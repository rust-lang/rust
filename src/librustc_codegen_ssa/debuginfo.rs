// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax_pos::{BytePos, Span};
use rustc::hir::def_id::CrateNum;
use std::cell::Cell;

pub enum FunctionDebugContext<D> {
    RegularContext(FunctionDebugContextData<D>),
    DebugInfoDisabled,
    FunctionWithoutDebugInfo,
}

impl<D> FunctionDebugContext<D> {
    pub fn get_ref<'a>(&'a self, span: Span) -> &'a FunctionDebugContextData<D> {
        match *self {
            FunctionDebugContext::RegularContext(ref data) => data,
            FunctionDebugContext::DebugInfoDisabled => {
                span_bug!(span, "{}", FunctionDebugContext::<D>::debuginfo_disabled_message());
            }
            FunctionDebugContext::FunctionWithoutDebugInfo => {
                span_bug!(span, "{}", FunctionDebugContext::<D>::should_be_ignored_message());
            }
        }
    }

    fn debuginfo_disabled_message() -> &'static str {
        "debuginfo: Error trying to access FunctionDebugContext although debug info is disabled!"
    }

    fn should_be_ignored_message() -> &'static str {
        "debuginfo: Error trying to access FunctionDebugContext for function that should be \
         ignored by debug info!"
    }
}

/// Enables emitting source locations for the given functions.
///
/// Since we don't want source locations to be emitted for the function prelude,
/// they are disabled when beginning to codegen a new function. This functions
/// switches source location emitting on and must therefore be called before the
/// first real statement/expression of the function is codegened.
pub fn start_emitting_source_locations<D>(dbg_context: &FunctionDebugContext<D>) {
    match *dbg_context {
        FunctionDebugContext::RegularContext(ref data) => {
            data.source_locations_enabled.set(true)
        },
        _ => { /* safe to ignore */ }
    }
}

pub struct FunctionDebugContextData<D> {
    fn_metadata: D,
    source_locations_enabled: Cell<bool>,
    pub defining_crate: CrateNum,
}

pub enum VariableAccess<'a, V> {
    // The llptr given is an alloca containing the variable's value
    DirectVariable { alloca: V },
    // The llptr given is an alloca containing the start of some pointer chain
    // leading to the variable's content.
    IndirectVariable { alloca: V, address_operations: &'a [i64] }
}

pub enum VariableKind {
    ArgumentVariable(usize /*index*/),
    LocalVariable,
}


#[derive(Clone, Copy, Debug)]
pub struct MirDebugScope<D> {
    pub scope_metadata: Option<D>,
    // Start and end offsets of the file to which this DIScope belongs.
    // These are used to quickly determine whether some span refers to the same file.
    pub file_start_pos: BytePos,
    pub file_end_pos: BytePos,
}

impl<D> MirDebugScope<D> {
    pub fn is_valid(&self) -> bool {
        !self.scope_metadata.is_none()
    }
}
