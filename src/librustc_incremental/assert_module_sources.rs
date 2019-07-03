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

use rustc::hir::def_id::LOCAL_CRATE;
use rustc::dep_graph::cgu_reuse_tracker::*;
use rustc::mir::mono::CodegenUnitNameBuilder;
use rustc::ty::TyCtxt;
use std::collections::BTreeSet;
use syntax::ast;
use syntax::symbol::{Symbol, sym};
use rustc::ich::{ATTR_PARTITION_REUSED, ATTR_PARTITION_CODEGENED,
                 ATTR_EXPECTED_CGU_REUSE};

const MODULE: Symbol = sym::module;
const CFG: Symbol = sym::cfg;
const KIND: Symbol = sym::kind;

pub fn assert_module_sources(tcx: TyCtxt<'_>) {
    tcx.dep_graph.with_ignore(|| {
        if tcx.sess.opts.incremental.is_none() {
            return;
        }

        let available_cgus = tcx
            .collect_and_partition_mono_items(LOCAL_CRATE)
            .1
            .iter()
            .map(|cgu| format!("{}", cgu.name()))
            .collect::<BTreeSet<String>>();

        let ams = AssertModuleSource {
            tcx,
            available_cgus
        };

        for attr in &tcx.hir().krate().attrs {
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
        let (expected_reuse, comp_kind) = if attr.check_name(ATTR_PARTITION_REUSED) {
            (CguReuse::PreLto, ComparisonKind::AtLeast)
        } else if attr.check_name(ATTR_PARTITION_CODEGENED) {
            (CguReuse::No, ComparisonKind::Exact)
        } else if attr.check_name(ATTR_EXPECTED_CGU_REUSE) {
            match &self.field(attr, KIND).as_str()[..] {
                "no" => (CguReuse::No, ComparisonKind::Exact),
                "pre-lto" => (CguReuse::PreLto, ComparisonKind::Exact),
                "post-lto" => (CguReuse::PostLto, ComparisonKind::Exact),
                "any" => (CguReuse::PreLto, ComparisonKind::AtLeast),
                other => {
                    self.tcx.sess.span_fatal(
                        attr.span,
                        &format!("unknown cgu-reuse-kind `{}` specified", other));
                }
            }
        } else {
            return;
        };

        if !self.tcx.sess.opts.debugging_opts.query_dep_graph {
            self.tcx.sess.span_fatal(
                attr.span,
                &format!("found CGU-reuse attribute but `-Zquery-dep-graph` \
                          was not specified"));
        }

        if !self.check_config(attr) {
            debug!("check_attr: config does not match, ignoring attr");
            return;
        }

        let user_path = self.field(attr, MODULE).as_str().to_string();
        let crate_name = self.tcx.crate_name(LOCAL_CRATE).as_str().to_string();

        if !user_path.starts_with(&crate_name) {
            let msg = format!("Found malformed codegen unit name `{}`. \
                Codegen units names must always start with the name of the \
                crate (`{}` in this case).", user_path, crate_name);
            self.tcx.sess.span_fatal(attr.span, &msg);
        }

        // Split of the "special suffix" if there is one.
        let (user_path, cgu_special_suffix) = if let Some(index) = user_path.rfind(".") {
            (&user_path[..index], Some(&user_path[index + 1 ..]))
        } else {
            (&user_path[..], None)
        };

        let mut cgu_path_components = user_path.split('-').collect::<Vec<_>>();

        // Remove the crate name
        assert_eq!(cgu_path_components.remove(0), crate_name);

        let cgu_name_builder = &mut CodegenUnitNameBuilder::new(self.tcx);
        let cgu_name = cgu_name_builder.build_cgu_name(LOCAL_CRATE,
                                                       cgu_path_components,
                                                       cgu_special_suffix);

        debug!("mapping '{}' to cgu name '{}'", self.field(attr, MODULE), cgu_name);

        if !self.available_cgus.contains(&cgu_name.as_str()[..]) {
            self.tcx.sess.span_err(attr.span,
                &format!("no module named `{}` (mangled: {}). \
                          Available modules: {}",
                    user_path,
                    cgu_name,
                    self.available_cgus
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")));
        }

        self.tcx.sess.cgu_reuse_tracker.set_expectation(&cgu_name.as_str(),
                                                        &user_path,
                                                        attr.span,
                                                        expected_reuse,
                                                        comp_kind);
    }

    fn field(&self, attr: &ast::Attribute, name: Symbol) -> ast::Name {
        for item in attr.meta_item_list().unwrap_or_else(Vec::new) {
            if item.check_name(name) {
                if let Some(value) = item.value_str() {
                    return value;
                } else {
                    self.tcx.sess.span_fatal(
                        item.span(),
                        &format!("associated value expected for `{}`", name));
                }
            }
        }

        self.tcx.sess.span_fatal(
            attr.span,
            &format!("no field `{}`", name));
    }

    /// Scan for a `cfg="foo"` attribute and check whether we have a
    /// cfg flag called `foo`.
    fn check_config(&self, attr: &ast::Attribute) -> bool {
        let config = &self.tcx.sess.parse_sess.config;
        let value = self.field(attr, CFG);
        debug!("check_config(config={:?}, value={:?})", config, value);
        if config.iter().any(|&(name, _)| name == value) {
            debug!("check_config: matched");
            return true;
        }
        debug!("check_config: no match found");
        return false;
    }
}
