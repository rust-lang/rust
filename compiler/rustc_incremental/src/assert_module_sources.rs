//! This pass is only used for UNIT TESTS related to incremental
//! compilation. It tests whether a particular `.o` file will be re-used
//! from a previous compilation or whether it must be regenerated.
//!
//! The user adds annotations to the crate of the following form:
//!
//! ```
//! #![rustc_partition_reused(module="spike", cfg="rpass2")]
//! #![rustc_partition_codegened(module="spike-x", cfg="rpass2")]
//! ```
//!
//! The first indicates (in the cfg `rpass2`) that `spike.o` will be
//! reused, the second that `spike-x.o` will be recreated. If these
//! annotations are inaccurate, errors are reported.
//!
//! The reason that we use `cfg=...` and not `#[cfg_attr]` is so that
//! the HIR doesn't change as a result of the annotations, which might
//! perturb the reuse results.
//!
//! `#![rustc_expected_cgu_reuse(module="spike", cfg="rpass2", kind="post-lto")]
//! allows for doing a more fine-grained check to see if pre- or post-lto data
//! was re-used.

use rustc_ast as ast;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::mir::mono::CodegenUnitNameBuilder;
use rustc_middle::ty::TyCtxt;
use rustc_session::cgu_reuse_tracker::*;
use rustc_span::symbol::{sym, Symbol};
use std::collections::BTreeSet;

pub fn assert_module_sources(tcx: TyCtxt<'_>) {
    tcx.dep_graph.with_ignore(|| {
        if tcx.sess.opts.incremental.is_none() {
            return;
        }

        let available_cgus = tcx
            .collect_and_partition_mono_items(LOCAL_CRATE)
            .1
            .iter()
            .map(|cgu| cgu.name().to_string())
            .collect::<BTreeSet<String>>();

        let ams = AssertModuleSource { tcx, available_cgus };

        for attr in tcx.hir().krate().item.attrs {
            ams.check_attr(attr);
        }
    })
}

struct AssertModuleSource<'tcx> {
    tcx: TyCtxt<'tcx>,
    available_cgus: BTreeSet<String>,
}

impl AssertModuleSource<'tcx> {
    fn check_attr(&self, attr: &ast::Attribute) {
        let (expected_reuse, comp_kind) =
            if self.tcx.sess.check_name(attr, sym::rustc_partition_reused) {
                (CguReuse::PreLto, ComparisonKind::AtLeast)
            } else if self.tcx.sess.check_name(attr, sym::rustc_partition_codegened) {
                (CguReuse::No, ComparisonKind::Exact)
            } else if self.tcx.sess.check_name(attr, sym::rustc_expected_cgu_reuse) {
                match self.field(attr, sym::kind) {
                    sym::no => (CguReuse::No, ComparisonKind::Exact),
                    sym::pre_dash_lto => (CguReuse::PreLto, ComparisonKind::Exact),
                    sym::post_dash_lto => (CguReuse::PostLto, ComparisonKind::Exact),
                    sym::any => (CguReuse::PreLto, ComparisonKind::AtLeast),
                    other => {
                        self.tcx.sess.span_fatal(
                            attr.span,
                            &format!("unknown cgu-reuse-kind `{}` specified", other),
                        );
                    }
                }
            } else {
                return;
            };

        if !self.tcx.sess.opts.debugging_opts.query_dep_graph {
            self.tcx.sess.span_fatal(
                attr.span,
                "found CGU-reuse attribute but `-Zquery-dep-graph` was not specified",
            );
        }

        if !self.check_config(attr) {
            debug!("check_attr: config does not match, ignoring attr");
            return;
        }

        let user_path = self.field(attr, sym::module).to_string();
        let crate_name = self.tcx.crate_name(LOCAL_CRATE).to_string();

        if !user_path.starts_with(&crate_name) {
            let msg = format!(
                "Found malformed codegen unit name `{}`. \
                Codegen units names must always start with the name of the \
                crate (`{}` in this case).",
                user_path, crate_name
            );
            self.tcx.sess.span_fatal(attr.span, &msg);
        }

        // Split of the "special suffix" if there is one.
        let (user_path, cgu_special_suffix) = if let Some(index) = user_path.rfind('.') {
            (&user_path[..index], Some(&user_path[index + 1..]))
        } else {
            (&user_path[..], None)
        };

        let mut iter = user_path.split('-');

        // Remove the crate name
        assert_eq!(iter.next().unwrap(), crate_name);

        let cgu_path_components = iter.collect::<Vec<_>>();

        let cgu_name_builder = &mut CodegenUnitNameBuilder::new(self.tcx);
        let cgu_name =
            cgu_name_builder.build_cgu_name(LOCAL_CRATE, cgu_path_components, cgu_special_suffix);

        debug!("mapping '{}' to cgu name '{}'", self.field(attr, sym::module), cgu_name);

        if !self.available_cgus.contains(&*cgu_name.as_str()) {
            self.tcx.sess.span_err(
                attr.span,
                &format!(
                    "no module named `{}` (mangled: {}). Available modules: {}",
                    user_path,
                    cgu_name,
                    self.available_cgus
                        .iter()
                        .map(|cgu| cgu.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            );
        }

        self.tcx.sess.cgu_reuse_tracker.set_expectation(
            cgu_name,
            &user_path,
            attr.span,
            expected_reuse,
            comp_kind,
        );
    }

    fn field(&self, attr: &ast::Attribute, name: Symbol) -> Symbol {
        for item in attr.meta_item_list().unwrap_or_else(Vec::new) {
            if item.has_name(name) {
                if let Some(value) = item.value_str() {
                    return value;
                } else {
                    self.tcx.sess.span_fatal(
                        item.span(),
                        &format!("associated value expected for `{}`", name),
                    );
                }
            }
        }

        self.tcx.sess.span_fatal(attr.span, &format!("no field `{}`", name));
    }

    /// Scan for a `cfg="foo"` attribute and check whether we have a
    /// cfg flag called `foo`.
    fn check_config(&self, attr: &ast::Attribute) -> bool {
        let config = &self.tcx.sess.parse_sess.config;
        let value = self.field(attr, sym::cfg);
        debug!("check_config(config={:?}, value={:?})", config, value);
        if config.iter().any(|&(name, _)| name == value) {
            debug!("check_config: matched");
            return true;
        }
        debug!("check_config: no match found");
        false
    }
}
