// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Walks the crate looking for items/impl-items/trait-items that have
//! either a `rustc_symbol_name` or `rustc_item_path` attribute and
//! generates an error giving, respectively, the symbol name or
//! item-path. This is used for unit testing the code that generates
//! paths etc in all kinds of annoying scenarios.

use base;
use rustc::hir;
use rustc::hir::def::Def;
use rustc::mir::mono::{Linkage, Visibility};
use rustc::ty::Ty;
use rustc::ty::layout::{LayoutOf, HasTyCtxt, TyLayout};
use std::fmt;
use interfaces::*;

pub use rustc::mir::mono::MonoItem;

pub use rustc_mir::monomorphize::item::MonoItemExt as BaseMonoItemExt;

pub trait MonoItemExt<'a, 'll: 'a, 'tcx: 'll> : fmt::Debug + BaseMonoItemExt<'ll, 'tcx>
{
    fn define<Bx: BuilderMethods<'a, 'll, 'tcx>>(&self, cx: &'a Bx::CodegenCx) where
        &'a Bx::CodegenCx : LayoutOf<Ty = Ty<'tcx>, TyLayout=TyLayout<'tcx>> + HasTyCtxt<'tcx>
    {
        debug!("BEGIN IMPLEMENTING '{} ({})' in cgu {}",
               self.to_string(*cx.tcx()),
               self.to_raw_string(),
               cx.codegen_unit().name());

        match *self.as_mono_item() {
            MonoItem::Static(def_id) => {
                let tcx = *cx.tcx();
                let is_mutable = match tcx.describe_def(def_id) {
                    Some(Def::Static(_, is_mutable)) => is_mutable,
                    Some(other) => {
                        bug!("Expected Def::Static, found {:?}", other)
                    }
                    None => {
                        bug!("Expected Def::Static for {:?}, found nothing", def_id)
                    }
                };
                cx.codegen_static(def_id, is_mutable);
            }
            MonoItem::GlobalAsm(node_id) => {
                let item = cx.tcx().hir.expect_item(node_id);
                if let hir::ItemKind::GlobalAsm(ref ga) = item.node {
                    cx.codegen_global_asm(ga);
                } else {
                    span_bug!(item.span, "Mismatch between hir::Item type and MonoItem type")
                }
            }
            MonoItem::Fn(instance) => {
                base::codegen_instance::<'a, 'll, 'tcx, Bx>(&cx, instance);
            }
        }

        debug!("END IMPLEMENTING '{} ({})' in cgu {}",
               self.to_string(*cx.tcx()),
               self.to_raw_string(),
               cx.codegen_unit().name());
    }

    fn predefine<Bx: BuilderMethods<'a, 'll, 'tcx>>(
        &self,
        cx: &'a Bx::CodegenCx,
        linkage: Linkage,
        visibility: Visibility
    ) where
        &'a Bx::CodegenCx : LayoutOf<Ty = Ty<'tcx>, TyLayout=TyLayout<'tcx>> + HasTyCtxt<'tcx>
    {
        debug!("BEGIN PREDEFINING '{} ({})' in cgu {}",
               self.to_string(*cx.tcx()),
               self.to_raw_string(),
               cx.codegen_unit().name());

        let symbol_name = self.symbol_name(*cx.tcx()).as_str();

        debug!("symbol {}", &symbol_name);

        match *self.as_mono_item() {
            MonoItem::Static(def_id) => {
                cx.predefine_static(def_id, linkage, visibility, &symbol_name);
            }
            MonoItem::Fn(instance) => {
                cx.predefine_fn(instance, linkage, visibility, &symbol_name);
            }
            MonoItem::GlobalAsm(..) => {}
        }

        debug!("END PREDEFINING '{} ({})' in cgu {}",
               self.to_string(*cx.tcx()),
               self.to_raw_string(),
               cx.codegen_unit().name());
    }

    fn to_raw_string(&self) -> String {
        match *self.as_mono_item() {
            MonoItem::Fn(instance) => {
                format!("Fn({:?}, {})",
                        instance.def,
                        instance.substs.as_ptr() as usize)
            }
            MonoItem::Static(id) => {
                format!("Static({:?})", id)
            }
            MonoItem::GlobalAsm(id) => {
                format!("GlobalAsm({:?})", id)
            }
        }
    }
}

impl<'a, 'll:'a, 'tcx: 'll> MonoItemExt<'a, 'll, 'tcx>
    for MonoItem<'tcx> {}
