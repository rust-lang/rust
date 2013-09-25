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

#[deriving(Clone, Eq)]
pub enum ArgKind {
    /// Pass the argument directly using the normal converted
    /// LLVM type or by coercing to another specified type
    Direct,
    /// Pass the argument indirectly via a hidden pointer
    Indirect
}

/// Information about how a specific C type
/// should be passed to or returned from a function
///
/// This is borrowed from clang's ABIInfo.h
#[deriving(Clone)]
pub struct ArgType {
    kind: ArgKind,
    /// Original LLVM type
    ty: Type,
    /// Coerced LLVM Type
    cast: option::Option<Type>,
    /// Dummy argument, which is emitted before the real argument
    pad: option::Option<Type>,
    /// LLVM attribute of argument
    attr: option::Option<Attribute>
}

impl ArgType {
    pub fn direct(ty: Type, cast: option::Option<Type>,
                            pad: option::Option<Type>,
                            attr: option::Option<Attribute>) -> ArgType {
        ArgType {
            kind: Direct,
            ty: ty,
            cast: cast,
            pad: pad,
            attr: attr
        }
    }

    pub fn indirect(ty: Type, attr: option::Option<Attribute>) -> ArgType {
        ArgType {
            kind: Indirect,
            ty: ty,
            cast: option::None,
            pad: option::None,
            attr: attr
        }
    }

    pub fn is_direct(&self) -> bool {
        return self.kind == Direct;
    }

    pub fn is_indirect(&self) -> bool {
        return self.kind == Indirect;
    }
}

/// Metadata describing how the arguments to a native function
/// should be passed in order to respect the native ABI.
///
/// I will do my best to describe this structure, but these
/// comments are reverse-engineered and may be inaccurate. -NDM
pub struct FnType {
    /// The LLVM types of each argument.
    arg_tys: ~[ArgType],

    /// LLVM return type.
    ret_ty: ArgType,
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
