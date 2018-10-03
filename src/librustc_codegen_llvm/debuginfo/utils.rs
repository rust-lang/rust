// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Utility Functions.

use super::{CrateDebugContext};
use super::namespace::item_namespace;

use rustc::hir::def_id::DefId;
use rustc::ty::DefIdTree;

use llvm;
use llvm::debuginfo::{DIScope, DIBuilder, DIDescriptor, DIArray};
use common::{CodegenCx};
use value::Value;
use rustc_codegen_ssa::interfaces::*;

use syntax_pos::{self, Span};

pub fn is_node_local_to_unit(cx: &CodegenCx<'ll, '_, &'ll Value>, def_id: DefId) -> bool
{
    // The is_local_to_unit flag indicates whether a function is local to the
    // current compilation unit (i.e. if it is *static* in the C-sense). The
    // *reachable* set should provide a good approximation of this, as it
    // contains everything that might leak out of the current crate (by being
    // externally visible or by being inlined into something externally
    // visible). It might better to use the `exported_items` set from
    // `driver::CrateAnalysis` in the future, but (atm) this set is not
    // available in the codegen pass.
    !cx.tcx.is_reachable_non_generic(def_id)
}

#[allow(non_snake_case)]
pub fn create_DIArray(
    builder: &DIBuilder<'ll>,
    arr: &[Option<&'ll DIDescriptor>],
) -> &'ll DIArray {
    return unsafe {
        llvm::LLVMRustDIBuilderGetOrCreateArray(builder, arr.as_ptr(), arr.len() as u32)
    };
}

/// Return syntax_pos::Loc corresponding to the beginning of the span
pub fn span_start(cx: &CodegenCx<'ll, '_, &'ll Value>, span: Span) -> syntax_pos::Loc {
    cx.sess().source_map().lookup_char_pos(span.lo())
}

#[inline]
pub fn debug_context(cx: &'a CodegenCx<'ll, 'tcx, &'ll Value>) -> &'a CrateDebugContext<'ll, 'tcx> {
    cx.dbg_cx.as_ref().unwrap()
}

#[inline]
#[allow(non_snake_case)]
pub fn DIB(cx: &'a CodegenCx<'ll, '_, &'ll Value>) -> &'a DIBuilder<'ll> {
    cx.dbg_cx.as_ref().unwrap().builder
}

pub fn get_namespace_for_item(cx: &CodegenCx<'ll, '_, &'ll Value>, def_id: DefId) -> &'ll DIScope {
    item_namespace(cx, cx.tcx.parent(def_id)
        .expect("get_namespace_for_item: missing parent?"))
}
