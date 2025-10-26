//! This pass is only used for UNIT TESTS related to incremental
//! compilation. It tests whether a particular `.o` file will be re-used
//! from a previous compilation or whether it must be regenerated.
//!
//! The user adds annotations to the crate of the following form:
//!
//! ```
//! # #![feature(rustc_attrs)]
//! # #![allow(internal_features)]
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
//! `#![rustc_expected_cgu_reuse(module="spike", cfg="rpass2", kind="post-lto")]`
//! allows for doing a more fine-grained check to see if pre- or post-lto data
//! was re-used.

use std::borrow::Cow;
use std::fmt;

use rustc_data_structures::unord::{UnordMap, UnordSet};
use rustc_errors::{DiagArgValue, IntoDiagArg};
use rustc_hir as hir;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::mir::mono::CodegenUnitNameBuilder;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_span::{Span, Symbol, sym};
use thin_vec::ThinVec;
use tracing::debug;

use crate::errors;

#[allow(missing_docs)]
pub fn assert_module_sources(tcx: TyCtxt<'_>, set_reuse: &dyn Fn(&mut CguReuseTracker)) {
    tcx.dep_graph.with_ignore(|| {
        if tcx.sess.opts.incremental.is_none() {
            return;
        }

        let available_cgus = tcx
            .collect_and_partition_mono_items(())
            .codegen_units
            .iter()
            .map(|cgu| cgu.name())
            .collect();

        let mut ams = AssertModuleSource {
            tcx,
            available_cgus,
            cgu_reuse_tracker: if tcx.sess.opts.unstable_opts.query_dep_graph {
                CguReuseTracker::new()
            } else {
                CguReuseTracker::new_disabled()
            },
        };

        for attr in tcx.hir_attrs(rustc_hir::CRATE_HIR_ID) {
            ams.check_attr(attr);
        }

        set_reuse(&mut ams.cgu_reuse_tracker);

        if tcx.sess.opts.unstable_opts.print_mono_items
            && let Some(data) = &ams.cgu_reuse_tracker.data
        {
            data.actual_reuse.items().all(|(cgu, reuse)| {
                println!("CGU_REUSE {cgu} {reuse}");
                true
            });
        }

        ams.cgu_reuse_tracker.check_expected_reuse(tcx.sess);
    });
}

struct AssertModuleSource<'tcx> {
    tcx: TyCtxt<'tcx>,
    available_cgus: UnordSet<Symbol>,
    cgu_reuse_tracker: CguReuseTracker,
}

impl<'tcx> AssertModuleSource<'tcx> {
    fn check_attr(&mut self, attr: &hir::Attribute) {
        let (expected_reuse, comp_kind) = if attr.has_name(sym::rustc_partition_reused) {
            (CguReuse::PreLto, ComparisonKind::AtLeast)
        } else if attr.has_name(sym::rustc_partition_codegened) {
            (CguReuse::No, ComparisonKind::Exact)
        } else if attr.has_name(sym::rustc_expected_cgu_reuse) {
            match self.field(attr, sym::kind) {
                sym::no => (CguReuse::No, ComparisonKind::Exact),
                sym::pre_dash_lto => (CguReuse::PreLto, ComparisonKind::Exact),
                sym::post_dash_lto => (CguReuse::PostLto, ComparisonKind::Exact),
                sym::any => (CguReuse::PreLto, ComparisonKind::AtLeast),
                other => {
                    self.tcx
                        .dcx()
                        .emit_fatal(errors::UnknownReuseKind { span: attr.span(), kind: other });
                }
            }
        } else {
            return;
        };

        if !self.tcx.sess.opts.unstable_opts.query_dep_graph {
            self.tcx.dcx().emit_fatal(errors::MissingQueryDepGraph { span: attr.span() });
        }

        if !self.check_config(attr) {
            debug!("check_attr: config does not match, ignoring attr");
            return;
        }

        let user_path = self.field(attr, sym::module).to_string();
        let crate_name = self.tcx.crate_name(LOCAL_CRATE).to_string();

        if !user_path.starts_with(&crate_name) {
            self.tcx.dcx().emit_fatal(errors::MalformedCguName {
                span: attr.span(),
                user_path,
                crate_name,
            });
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

        if !self.available_cgus.contains(&cgu_name) {
            let cgu_names: Vec<&str> =
                self.available_cgus.items().map(|cgu| cgu.as_str()).into_sorted_stable_ord();
            self.tcx.dcx().emit_err(errors::NoModuleNamed {
                span: attr.span(),
                user_path,
                cgu_name,
                cgu_names: cgu_names.join(", "),
            });
        }

        self.cgu_reuse_tracker.set_expectation(
            cgu_name,
            user_path,
            attr.span(),
            expected_reuse,
            comp_kind,
        );
    }

    fn field(&self, attr: &hir::Attribute, name: Symbol) -> Symbol {
        for item in attr.meta_item_list().unwrap_or_else(ThinVec::new) {
            if item.has_name(name) {
                if let Some(value) = item.value_str() {
                    return value;
                } else {
                    self.tcx.dcx().emit_fatal(errors::FieldAssociatedValueExpected {
                        span: item.span(),
                        name,
                    });
                }
            }
        }

        self.tcx.dcx().emit_fatal(errors::NoField { span: attr.span(), name });
    }

    /// Scan for a `cfg="foo"` attribute and check whether we have a
    /// cfg flag called `foo`.
    fn check_config(&self, attr: &hir::Attribute) -> bool {
        let config = &self.tcx.sess.psess.config;
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

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum CguReuse {
    No,
    PreLto,
    PostLto,
}

impl fmt::Display for CguReuse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            CguReuse::No => write!(f, "No"),
            CguReuse::PreLto => write!(f, "PreLto"),
            CguReuse::PostLto => write!(f, "PostLto"),
        }
    }
}

impl IntoDiagArg for CguReuse {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(self.to_string()))
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ComparisonKind {
    Exact,
    AtLeast,
}

struct TrackerData {
    actual_reuse: UnordMap<String, CguReuse>,
    expected_reuse: UnordMap<String, (String, Span, CguReuse, ComparisonKind)>,
}

pub struct CguReuseTracker {
    data: Option<TrackerData>,
}

impl CguReuseTracker {
    fn new() -> CguReuseTracker {
        let data =
            TrackerData { actual_reuse: Default::default(), expected_reuse: Default::default() };

        CguReuseTracker { data: Some(data) }
    }

    fn new_disabled() -> CguReuseTracker {
        CguReuseTracker { data: None }
    }

    pub fn set_actual_reuse(&mut self, cgu_name: &str, kind: CguReuse) {
        if let Some(data) = &mut self.data {
            debug!("set_actual_reuse({cgu_name:?}, {kind:?})");

            let prev_reuse = data.actual_reuse.insert(cgu_name.to_string(), kind);
            assert!(prev_reuse.is_none());
        }
    }

    fn set_expectation(
        &mut self,
        cgu_name: Symbol,
        cgu_user_name: &str,
        error_span: Span,
        expected_reuse: CguReuse,
        comparison_kind: ComparisonKind,
    ) {
        if let Some(data) = &mut self.data {
            debug!("set_expectation({cgu_name:?}, {expected_reuse:?}, {comparison_kind:?})");

            data.expected_reuse.insert(
                cgu_name.to_string(),
                (cgu_user_name.to_string(), error_span, expected_reuse, comparison_kind),
            );
        }
    }

    fn check_expected_reuse(&self, sess: &Session) {
        if let Some(ref data) = self.data {
            let keys = data.expected_reuse.keys().into_sorted_stable_ord();
            for cgu_name in keys {
                let &(ref cgu_user_name, ref error_span, expected_reuse, comparison_kind) =
                    data.expected_reuse.get(cgu_name).unwrap();

                if let Some(&actual_reuse) = data.actual_reuse.get(cgu_name) {
                    let (error, at_least) = match comparison_kind {
                        ComparisonKind::Exact => (expected_reuse != actual_reuse, false),
                        ComparisonKind::AtLeast => (actual_reuse < expected_reuse, true),
                    };

                    if error {
                        let at_least = if at_least { 1 } else { 0 };
                        sess.dcx().emit_err(errors::IncorrectCguReuseType {
                            span: *error_span,
                            cgu_user_name,
                            actual_reuse,
                            expected_reuse,
                            at_least,
                        });
                    }
                } else {
                    sess.dcx().emit_fatal(errors::CguNotRecorded { cgu_user_name, cgu_name });
                }
            }
        }
    }
}
