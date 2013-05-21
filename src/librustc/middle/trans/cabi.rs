// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lib::llvm::Attribute;
use std::option;
use middle::trans::context::CrateContext;
use middle::trans::cabi_x86;
use middle::trans::cabi_x86_64;
use middle::trans::cabi_arm;
use middle::trans::cabi_mips;
use middle::trans::type_::Type;
use syntax::abi::{X86, X86_64, Arm, Mips};

#[deriving(Clone)]
pub struct LLVMType {
    cast: bool,
    ty: Type
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

pub fn compute_abi_info(ccx: &mut CrateContext,
                        atys: &[Type],
                        rty: Type,
                        ret_def: bool) -> FnType {
    match ccx.sess.targ_cfg.arch {
        X86 => cabi_x86::compute_abi_info(ccx, atys, rty, ret_def),
        X86_64 => cabi_x86_64::compute_abi_info(ccx, atys, rty, ret_def),
        Arm => cabi_arm::compute_abi_info(ccx, atys, rty, ret_def),
        Mips => cabi_mips::compute_abi_info(ccx, atys, rty, ret_def),
    }
}
