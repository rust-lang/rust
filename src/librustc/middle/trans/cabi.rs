// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lib::llvm::{TypeRef, Attribute};
use core::option;

pub trait ABIInfo {
    fn compute_info(&self,
                    atys: &[TypeRef],
                    rty: TypeRef,
                    ret_def: bool) -> FnType;
}

pub struct LLVMType {
    cast: bool,
    ty: TypeRef
}

/// Metadata describing how the arguments to a native function
/// should be passed in order to respect the native ABI.
///
/// I will do my best to describe this structure, but these
/// comments are reverse-engineered and may be inaccurate. -NDM
pub struct FnType {
    /// The LLVM types of each argument. If the cast flag is true,
    /// then the argument should be cast, typically because the
    /// official argument type will be an int and the rust type is i8
    /// or something like that.
    arg_tys: ~[LLVMType],

    /// A list of attributes to be attached to each argument (parallel
    /// the `arg_tys` array). If the attribute for a given is Some,
    /// then the argument should be passed by reference.
    attrs: ~[option::Option<Attribute>],

    /// LLVM return type.
    ret_ty: LLVMType,

    /// If true, then an implicit pointer should be added for the result.
    sret: bool
}
