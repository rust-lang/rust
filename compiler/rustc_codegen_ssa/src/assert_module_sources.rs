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
use rustc_hir::attrs::{AttributeKind, CguFields, CguKind};
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_hir::{self as hir, find_attr};
use rustc_middle::mir::mono::CodegenUnitNameBuilder;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_span::{Span, Symbol};
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

        ams.check_attrs(tcx.hir_attrs(rustc_hir::CRATE_HIR_ID));

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
    fn check_attrs(&mut self, attrs: &[hir::Attribute]) {
        for &(span, cgu_fields) in find_attr!(attrs,
            AttributeKind::RustcCguTestAttr(e) => e)
        .into_iter()
        .flatten()
        {
            let (expected_reuse, comp_kind) = match cgu_fields {
                CguFields::PartitionReused { .. } => (CguReuse::PreLto, ComparisonKind::AtLeast),
                CguFields::PartitionCodegened { .. } => (CguReuse::No, ComparisonKind::Exact),
                CguFields::ExpectedCguReuse { kind, .. } => match kind {
                    CguKind::No => (CguReuse::No, ComparisonKind::Exact),
                    CguKind::PreDashLto => (CguReuse::PreLto, ComparisonKind::Exact),
                    CguKind::PostDashLto => (CguReuse::PostLto, ComparisonKind::Exact),
                    CguKind::Any => (CguReuse::PreLto, ComparisonKind::AtLeast),
                },
            };
            let (CguFields::ExpectedCguReuse { cfg, module, .. }
            | CguFields::PartitionCodegened { cfg, module }
            | CguFields::PartitionReused { cfg, module }) = cgu_fields;

            if !self.tcx.sess.opts.unstable_opts.query_dep_graph {
                self.tcx.dcx().emit_fatal(errors::MissingQueryDepGraph { span });
            }

            if !self.check_config(cfg) {
                debug!("check_attr: config does not match, ignoring attr");
                return;
            }

            let user_path = module.as_str();
            let crate_name = self.tcx.crate_name(LOCAL_CRATE);
            let crate_name = crate_name.as_str();

            if !user_path.starts_with(&crate_name) {
                self.tcx.dcx().emit_fatal(errors::MalformedCguName { span, user_path, crate_name });
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
            let cgu_name = cgu_name_builder.build_cgu_name(
                LOCAL_CRATE,
                cgu_path_components,
                cgu_special_suffix,
            );

            debug!("mapping '{user_path}' to cgu name '{cgu_name}'");

            if !self.available_cgus.contains(&cgu_name) {
                let cgu_names: Vec<&str> =
                    self.available_cgus.items().map(|cgu| cgu.as_str()).into_sorted_stable_ord();
                self.tcx.dcx().emit_err(errors::NoModuleNamed {
                    span,
                    user_path,
                    cgu_name,
                    cgu_names: cgu_names.join(", "),
                });
            }

            self.cgu_reuse_tracker.set_expectation(
                cgu_name,
                user_path,
                span,
                expected_reuse,
                comp_kind,
            );
        }
    }

    /// Scan for a `cfg="foo"` attribute and check whether we have a
    /// cfg flag called `foo`.
    fn check_config(&self, value: Symbol) -> bool {
        let config = &self.tcx.sess.psess.config;
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
