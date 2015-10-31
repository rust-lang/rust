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

use super::{FunctionDebugContext, CrateDebugContext};
use super::namespace::namespace_for_item;

use middle::def_id::DefId;

use llvm;
use llvm::debuginfo::{DIScope, DIBuilderRef, DIDescriptor, DIArray};
use trans::machine;
use trans::common::{CrateContext, FunctionContext};
use trans::type_::Type;

use syntax::codemap::Span;
use syntax::{ast, codemap};

pub fn is_node_local_to_unit(cx: &CrateContext, node_id: ast::NodeId) -> bool
{
    // The is_local_to_unit flag indicates whether a function is local to the
    // current compilation unit (i.e. if it is *static* in the C-sense). The
    // *reachable* set should provide a good approximation of this, as it
    // contains everything that might leak out of the current crate (by being
    // externally visible or by being inlined into something externally
    // visible). It might better to use the `exported_items` set from
    // `driver::CrateAnalysis` in the future, but (atm) this set is not
    // available in the translation pass.
    !cx.reachable().contains(&node_id)
}

#[allow(non_snake_case)]
pub fn create_DIArray(builder: DIBuilderRef, arr: &[DIDescriptor]) -> DIArray {
    return unsafe {
        llvm::LLVMDIBuilderGetOrCreateArray(builder, arr.as_ptr(), arr.len() as u32)
    };
}

pub fn contains_nodebug_attribute(attributes: &[ast::Attribute]) -> bool {
    attributes.iter().any(|attr| {
        let meta_item: &ast::MetaItem = &*attr.node.value;
        match meta_item.node {
            ast::MetaWord(ref value) => &value[..] == "no_debug",
            _ => false
        }
    })
}

/// Return codemap::Loc corresponding to the beginning of the span
pub fn span_start(cx: &CrateContext, span: Span) -> codemap::Loc {
    cx.sess().codemap().lookup_char_pos(span.lo)
}

pub fn size_and_align_of(cx: &CrateContext, llvm_type: Type) -> (u64, u64) {
    (machine::llsize_of_alloc(cx, llvm_type), machine::llalign_of_min(cx, llvm_type) as u64)
}

pub fn bytes_to_bits(bytes: u64) -> u64 {
    bytes * 8
}

#[inline]
pub fn debug_context<'a, 'tcx>(cx: &'a CrateContext<'a, 'tcx>)
                           -> &'a CrateDebugContext<'tcx> {
    let debug_context: &'a CrateDebugContext<'tcx> = cx.dbg_cx().as_ref().unwrap();
    debug_context
}

#[inline]
#[allow(non_snake_case)]
pub fn DIB(cx: &CrateContext) -> DIBuilderRef {
    cx.dbg_cx().as_ref().unwrap().builder
}

pub fn fn_should_be_ignored(fcx: &FunctionContext) -> bool {
    match fcx.debug_context {
        FunctionDebugContext::RegularContext(_) => false,
        _ => true
    }
}

pub fn assert_type_for_node_id(cx: &CrateContext,
                           node_id: ast::NodeId,
                           error_reporting_span: Span) {
    if !cx.tcx().node_types().contains_key(&node_id) {
        cx.sess().span_bug(error_reporting_span,
                           "debuginfo: Could not find type for node id!");
    }
}

pub fn get_namespace_and_span_for_item(cx: &CrateContext, def_id: DefId)
                                   -> (DIScope, Span) {
    let containing_scope = namespace_for_item(cx, def_id).scope;
    let definition_span = cx.tcx().map.def_id_span(def_id, codemap::DUMMY_SP /* (1) */ );

    // (1) For external items there is no span information

    (containing_scope, definition_span)
}
