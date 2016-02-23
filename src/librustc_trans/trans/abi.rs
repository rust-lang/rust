// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::ArgKind::*;

use llvm::{self, AttrHelper, ValueRef};
use trans::attributes;
use trans::common::return_type_is_void;
use trans::context::CrateContext;
use trans::cabi_x86;
use trans::cabi_x86_64;
use trans::cabi_x86_win64;
use trans::cabi_arm;
use trans::cabi_aarch64;
use trans::cabi_powerpc;
use trans::cabi_powerpc64;
use trans::cabi_mips;
use trans::cabi_asmjs;
use trans::type_::Type;
use trans::type_of;

use middle::ty::{self, Ty};

pub use syntax::abi::Abi;

/// The first half of a fat pointer.
/// - For a closure, this is the code address.
/// - For an object or trait instance, this is the address of the box.
/// - For a slice, this is the base address.
pub const FAT_PTR_ADDR: usize = 0;

/// The second half of a fat pointer.
/// - For a closure, this is the address of the environment.
/// - For an object or trait instance, this is the address of the vtable.
/// - For a slice, this is the length.
pub const FAT_PTR_EXTRA: usize = 1;

#[derive(Clone, Copy, PartialEq, Debug)]
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
#[derive(Clone, Copy, Debug)]
pub struct ArgType {
    pub kind: ArgKind,
    /// Original LLVM type
    pub ty: Type,
    /// Coerced LLVM Type
    pub cast: Option<Type>,
    /// Dummy argument, which is emitted before the real argument
    pub pad: Option<Type>,
    /// LLVM attribute of argument
    pub attr: Option<llvm::Attribute>
}

impl ArgType {
    pub fn direct(ty: Type, cast: Option<Type>,
                            pad: Option<Type>,
                            attr: Option<llvm::Attribute>) -> ArgType {
        ArgType {
            kind: Direct,
            ty: ty,
            cast: cast,
            pad: pad,
            attr: attr
        }
    }

    pub fn indirect(ty: Type, attr: Option<llvm::Attribute>) -> ArgType {
        ArgType {
            kind: Indirect,
            ty: ty,
            cast: Option::None,
            pad: Option::None,
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

fn c_type_of<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>, ty: Ty<'tcx>) -> Type {
    if ty.is_bool() {
        Type::i1(cx)
    } else {
        type_of::type_of(cx, ty)
    }
}

/// Metadata describing how the arguments to a native function
/// should be passed in order to respect the native ABI.
///
/// I will do my best to describe this structure, but these
/// comments are reverse-engineered and may be inaccurate. -NDM
pub struct FnType {
    /// The LLVM types of each argument.
    pub args: Vec<ArgType>,

    /// LLVM return type.
    pub ret: ArgType,

    pub variadic: bool,

    pub cconv: llvm::CallConv
}

impl FnType {
    pub fn new<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                         abi: Abi,
                         sig: &ty::FnSig<'tcx>,
                         extra_args: &[Ty<'tcx>]) -> FnType {
        use self::Abi::*;
        let cconv = match ccx.sess().target.target.adjust_abi(abi) {
            RustIntrinsic => {
                // Intrinsics are emitted at the call site
                ccx.sess().bug("asked to register intrinsic fn");
            }
            PlatformIntrinsic => {
                // Intrinsics are emitted at the call site
                ccx.sess().bug("asked to register platform intrinsic fn");
            }

            Rust => {
                // FIXME(#3678) Implement linking to foreign fns with Rust ABI
                ccx.sess().unimpl("foreign functions with Rust ABI");
            }

            RustCall => {
                // FIXME(#3678) Implement linking to foreign fns with Rust ABI
                ccx.sess().unimpl("foreign functions with RustCall ABI");
            }

            // It's the ABI's job to select this, not us.
            System => ccx.sess().bug("system abi should be selected elsewhere"),

            Stdcall => llvm::X86StdcallCallConv,
            Fastcall => llvm::X86FastcallCallConv,
            Vectorcall => llvm::X86_VectorCall,
            C => llvm::CCallConv,
            Win64 => llvm::X86_64_Win64,

            // These API constants ought to be more specific...
            Cdecl => llvm::CCallConv,
            Aapcs => llvm::CCallConv,
        };

        let rty = match sig.output {
            ty::FnConverging(ret_ty) if !return_type_is_void(ccx, ret_ty) => {
                c_type_of(ccx, ret_ty)
            }
            _ => Type::void(ccx)
        };

        let mut fty = FnType {
            args: sig.inputs.iter().chain(extra_args.iter()).map(|&ty| {
                ArgType::direct(c_type_of(ccx, ty), None, None, None)
            }).collect(),
            ret: ArgType::direct(rty, None, None, None),
            variadic: sig.variadic,
            cconv: cconv
        };

        match &ccx.sess().target.target.arch[..] {
            "x86" => cabi_x86::compute_abi_info(ccx, &mut fty),
            "x86_64" => if ccx.sess().target.target.options.is_like_windows {
                cabi_x86_win64::compute_abi_info(ccx, &mut fty);
            } else {
                cabi_x86_64::compute_abi_info(ccx, &mut fty);
            },
            "aarch64" => cabi_aarch64::compute_abi_info(ccx, &mut fty),
            "arm" => {
                let flavor = if ccx.sess().target.target.target_os == "ios" {
                    cabi_arm::Flavor::Ios
                } else {
                    cabi_arm::Flavor::General
                };
                cabi_arm::compute_abi_info(ccx, &mut fty, flavor);
            },
            "mips" => cabi_mips::compute_abi_info(ccx, &mut fty),
            "powerpc" => cabi_powerpc::compute_abi_info(ccx, &mut fty),
            "powerpc64" => cabi_powerpc64::compute_abi_info(ccx, &mut fty),
            "asmjs" => cabi_asmjs::compute_abi_info(ccx, &mut fty),
            a => ccx.sess().fatal(&format!("unrecognized arch \"{}\" in target specification", a))
        }

        fty
    }

    pub fn to_llvm(&self, ccx: &CrateContext) -> Type {
        let mut llargument_tys = Vec::new();

        let llreturn_ty = if self.ret.is_indirect() {
            llargument_tys.push(self.ret.ty.ptr_to());
            Type::void(ccx)
        } else {
            self.ret.cast.unwrap_or(self.ret.ty)
        };

        for arg in &self.args {
            if arg.is_ignore() {
                continue;
            }
            // add padding
            if let Some(ty) = arg.pad {
                llargument_tys.push(ty);
            }

            let llarg_ty = if arg.is_indirect() {
                arg.ty.ptr_to()
            } else {
                arg.cast.unwrap_or(arg.ty)
            };

            llargument_tys.push(llarg_ty);
        }

        if self.variadic {
            Type::variadic_func(&llargument_tys, &llreturn_ty)
        } else {
            Type::func(&llargument_tys, &llreturn_ty)
        }
    }

    pub fn add_attributes(&self, llfn: ValueRef) {
        let mut i = if self.ret.is_indirect() {
            1
        } else {
            0
        };

        if let Some(attr) = self.ret.attr {
            attr.apply_llfn(i, llfn);
        }

        i += 1;

        for arg in &self.args {
            if arg.is_ignore() {
                continue;
            }
            // skip padding
            if arg.pad.is_some() { i += 1; }

            if let Some(attr) = arg.attr {
                attr.apply_llfn(i, llfn);
            }

            i += 1;
        }

        attributes::unwind(llfn, false);
    }
}
