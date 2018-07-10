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

use rustc::hir::def_id::LOCAL_CRATE;
use rustc::dep_graph::{DepNode, DepConstructor};
use rustc::mir::mono::CodegenUnit;
use rustc::ty::TyCtxt;
use syntax::ast;
use rustc::ich::{ATTR_PARTITION_REUSED, ATTR_PARTITION_CODEGENED};

const MODULE: &'static str = "module";
const CFG: &'static str = "cfg";

#[derive(Debug, PartialEq, Clone, Copy)]
enum Disposition { Reused, Codegened }

pub fn assert_module_sources<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
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
        } else if attr.check_name(ATTR_PARTITION_CODEGENED) {
            Disposition::Codegened
        } else {
            return;
        };

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

        let mut cgu_path_components = user_path.split("-").collect::<Vec<_>>();

        // Remove the crate name
        assert_eq!(cgu_path_components.remove(0), crate_name);

        let cgu_name = CodegenUnit::build_cgu_name(self.tcx,
                                                   LOCAL_CRATE,
                                                   cgu_path_components,
                                                   cgu_special_suffix);

        debug!("mapping '{}' to cgu name '{}'", self.field(attr, MODULE), cgu_name);

        let dep_node = DepNode::new(self.tcx,
                                    DepConstructor::CompileCodegenUnit(cgu_name));

        if let Some(loaded_from_cache) = self.tcx.dep_graph.was_loaded_from_cache(&dep_node) {
            match (disposition, loaded_from_cache) {
                (Disposition::Reused, false) => {
                    self.tcx.sess.span_err(
                        attr.span,
                        &format!("expected module named `{}` to be Reused but is Codegened",
                                 user_path));
                }
                (Disposition::Codegened, true) => {
                    self.tcx.sess.span_err(
                        attr.span,
                        &format!("expected module named `{}` to be Codegened but is Reused",
                                 user_path));
                }
                (Disposition::Reused, true) |
                (Disposition::Codegened, false) => {
                    // These are what we would expect.
                }
            }
        } else {
            let available_cgus = self.tcx
                .collect_and_partition_mono_items(LOCAL_CRATE)
                .1
                .iter()
                .map(|cgu| format!("{}", cgu.name()))
                .collect::<Vec<String>>()
                .join(", ");

            self.tcx.sess.span_err(attr.span,
                &format!("no module named `{}` (mangled: {}).\nAvailable modules: {}",
                    user_path,
                    cgu_name,
                    available_cgus));
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
