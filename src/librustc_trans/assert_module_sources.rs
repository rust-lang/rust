// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This pass is only used for UNIT TESTS related to incremental
//! compilation. It tests whether a particular `.o` file will be re-used
//! from a previous compilation or whether it must be regenerated.
//!
//! The user adds annotations to the crate of the following form:
//!
//! ```
//! #![rustc_partition_reused(module="spike", cfg="rpass2")]
//! #![rustc_partition_translated(module="spike-x", cfg="rpass2")]
//! ```
//!
//! The first indicates (in the cfg `rpass2`) that `spike.o` will be
//! reused, the second that `spike-x.o` will be recreated. If these
//! annotations are inaccurate, errors are reported.
//!
//! The reason that we use `cfg=...` and not `#[cfg_attr]` is so that
//! the HIR doesn't change as a result of the annotations, which might
//! perturb the reuse results.

use rustc::ty::TyCtxt;
use syntax::ast;

use {ModuleSource, ModuleTranslation};

const PARTITION_REUSED: &'static str = "rustc_partition_reused";
const PARTITION_TRANSLATED: &'static str = "rustc_partition_translated";

const MODULE: &'static str = "module";
const CFG: &'static str = "cfg";

#[derive(Debug, PartialEq)]
enum Disposition { Reused, Translated }

pub fn assert_module_sources<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                       modules: &[ModuleTranslation]) {
    let _ignore = tcx.dep_graph.in_ignore();

    if tcx.sess.opts.incremental.is_none() {
        return;
    }

    let ams = AssertModuleSource { tcx: tcx, modules: modules };
    for attr in &tcx.map.krate().attrs {
        ams.check_attr(attr);
    }
}

struct AssertModuleSource<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    modules: &'a [ModuleTranslation],
}

impl<'a, 'tcx> AssertModuleSource<'a, 'tcx> {
    fn check_attr(&self, attr: &ast::Attribute) {
        let disposition = if attr.check_name(PARTITION_REUSED) {
            Disposition::Reused
        } else if attr.check_name(PARTITION_TRANSLATED) {
            Disposition::Translated
        } else {
            return;
        };

        if !self.check_config(attr) {
            debug!("check_attr: config does not match, ignoring attr");
            return;
        }

        let mname = self.field(attr, MODULE);
        let mtrans = self.modules.iter().find(|mtrans| *mtrans.name == *mname.as_str());
        let mtrans = match mtrans {
            Some(m) => m,
            None => {
                debug!("module name `{}` not found amongst:", mname);
                for mtrans in self.modules {
                    debug!("module named `{}` with disposition {:?}",
                           mtrans.name,
                           self.disposition(mtrans));
                }

                self.tcx.sess.span_err(
                    attr.span,
                    &format!("no module named `{}`", mname));
                return;
            }
        };

        let mtrans_disposition = self.disposition(mtrans);
        if disposition != mtrans_disposition {
            self.tcx.sess.span_err(
                attr.span,
                &format!("expected module named `{}` to be {:?} but is {:?}",
                         mname,
                         disposition,
                         mtrans_disposition));
        }
    }

    fn disposition(&self, mtrans: &ModuleTranslation) -> Disposition {
        match mtrans.source {
            ModuleSource::Preexisting(_) => Disposition::Reused,
            ModuleSource::Translated(_) => Disposition::Translated,
        }
    }

    fn field(&self, attr: &ast::Attribute, name: &str) -> ast::Name {
        for item in attr.meta_item_list().unwrap_or(&[]) {
            if item.check_name(name) {
                if let Some(value) = item.value_str() {
                    return value;
                } else {
                    self.tcx.sess.span_fatal(
                        item.span,
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
