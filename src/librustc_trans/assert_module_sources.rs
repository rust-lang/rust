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

use rustc::dep_graph::{DepNode, DepConstructor};
use rustc::mir::mono::CodegenUnit;
use rustc::ty::TyCtxt;
use syntax::ast;
use syntax_pos::symbol::Symbol;
use rustc::ich::{ATTR_PARTITION_REUSED, ATTR_PARTITION_TRANSLATED};

const MODULE: &'static str = "module";
const CFG: &'static str = "cfg";

#[derive(Debug, PartialEq, Clone, Copy)]
enum Disposition { Reused, Translated }

pub(crate) fn assert_module_sources<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    tcx.dep_graph.with_ignore(|| {
        if tcx.sess.opts.incremental.is_none() {
            return;
        }

        let ams = AssertModuleSource { tcx };
        for attr in &tcx.hir.krate().attrs {
            ams.check_attr(attr);
        }
    })
}

struct AssertModuleSource<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>
}

impl<'a, 'tcx> AssertModuleSource<'a, 'tcx> {
    fn check_attr(&self, attr: &ast::Attribute) {
        let disposition = if attr.check_name(ATTR_PARTITION_REUSED) {
            Disposition::Reused
        } else if attr.check_name(ATTR_PARTITION_TRANSLATED) {
            Disposition::Translated
        } else {
            return;
        };

        if !self.check_config(attr) {
            debug!("check_attr: config does not match, ignoring attr");
            return;
        }

        let mname = self.field(attr, MODULE);
        let mangled_cgu_name = CodegenUnit::mangle_name(&mname.as_str());
        let mangled_cgu_name = Symbol::intern(&mangled_cgu_name).as_str();

        let dep_node = DepNode::new(self.tcx,
                                    DepConstructor::CompileCodegenUnit(mangled_cgu_name));

        if let Some(loaded_from_cache) = self.tcx.dep_graph.was_loaded_from_cache(&dep_node) {
            match (disposition, loaded_from_cache) {
                (Disposition::Reused, false) => {
                    self.tcx.sess.span_err(
                        attr.span,
                        &format!("expected module named `{}` to be Reused but is Translated",
                                 mname));
                }
                (Disposition::Translated, true) => {
                    self.tcx.sess.span_err(
                        attr.span,
                        &format!("expected module named `{}` to be Translated but is Reused",
                                 mname));
                }
                (Disposition::Reused, true) |
                (Disposition::Translated, false) => {
                    // These are what we would expect.
                }
            }
        } else {
            self.tcx.sess.span_err(attr.span, &format!("no module named `{}`", mname));
        }
    }

    fn field(&self, attr: &ast::Attribute, name: &str) -> ast::Name {
        for item in attr.meta_item_list().unwrap_or_else(Vec::new) {
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
