// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::ArgKind::*;

use llvm::Attribute;
use std::option;
use trans::context::CrateContext;
use trans::cabi_x86;
use trans::cabi_x86_64;
use trans::cabi_x86_win64;
use trans::cabi_arm;
use trans::cabi_aarch64;
use trans::cabi_powerpc;
use trans::cabi_powerpc64;
use trans::cabi_mips;
use trans::type_::Type;

#[derive(Clone, Copy, PartialEq)]
pub enum ArgKind {
    /// Pass the argument directly using the normal converted
    /// LLVM type or by coercing to another specified type
    Direct,
    /// Pass the argument indirectly via a hidden pointer
    Indirect,
    /// Ignore the argument (useful for empty struct)
    Ignore,
}

/// Information about how a specific C type
/// should be passed to or returned from a function
///
/// This is borrowed from clang's ABIInfo.h
#[derive(Clone, Copy)]
pub struct ArgType {
    pub kind: ArgKind,
    /// Original LLVM type
    pub ty: Type,
    /// Coerced LLVM Type
    pub cast: option::Option<Type>,
    /// Dummy argument, which is emitted before the real argument
    pub pad: option::Option<Type>,
    /// LLVM attribute of argument
    pub attr: option::Option<Attribute>
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
            cast: option::Option::None,
            pad: option::Option::None,
            attr: attr
        }
    }

    pub fn ignore(ty: Type) -> ArgType {
        ArgType {
            kind: Ignore,
            ty: ty,
            cast: None,
            pad: None,
            attr: None,
        }
    }

    pub fn is_indirect(&self) -> bool {
        return self.kind == Indirect;
    }

    pub fn is_ignore(&self) -> bool {
        return self.kind == Ignore;
    }
}

/// Metadata describing how the arguments to a native function
/// should be passed in order to respect the native ABI.
///
/// I will do my best to describe this structure, but these
/// comments are reverse-engineered and may be inaccurate. -NDM
pub struct FnType {
    /// The LLVM types of each argument.
    pub arg_tys: Vec<ArgType> ,

    /// LLVM return type.
    pub ret_ty: ArgType,
}

pub fn compute_abi_info(ccx: &CrateContext,
                        atys: &[Type],
                        rty: Type,
                        ret_def: bool) -> FnType {
    match &ccx.sess().target.target.arch[..] {
        "x86" => cabi_x86::compute_abi_info(ccx, atys, rty, ret_def),
        "x86_64" => if ccx.sess().target.target.options.is_like_windows {
            cabi_x86_win64::compute_abi_info(ccx, atys, rty, ret_def)
        } else {
            cabi_x86_64::compute_abi_info(ccx, atys, rty, ret_def)
        },
        "aarch64" => cabi_aarch64::compute_abi_info(ccx, atys, rty, ret_def),
        "arm" => {
            let flavor = if ccx.sess().target.target.target_os == "ios" {
                cabi_arm::Flavor::Ios
            } else {
                cabi_arm::Flavor::General
            };
            cabi_arm::compute_abi_info(ccx, atys, rty, ret_def, flavor)
        },
        "mips" => cabi_mips::compute_abi_info(ccx, atys, rty, ret_def),
        "powerpc" => cabi_powerpc::compute_abi_info(ccx, atys, rty, ret_def),
        "powerpc64" | "powerpc64le" => cabi_powerpc64::compute_abi_info(ccx, atys, rty, ret_def),
        a => ccx.sess().fatal(&format!("unrecognized arch \"{}\" in target specification", a)
                              ),
    }
}
