// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lib::llvm::{llvm, ValueRef, Attribute, Void};
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::common::*;

use middle::trans::type_::Type;

use std::libc::c_uint;
use std::option;

pub trait ABIInfo {
    fn compute_info(&self, atys: &[Type], rty: Type, ret_def: bool) -> FnType;
}

#[deriving(Clone)]
pub struct LLVMType {
    cast: bool,
    ty: Type
}

pub struct FnType {
    arg_tys: ~[LLVMType],
    ret_ty: LLVMType,
    attrs: ~[option::Option<Attribute>],
    sret: bool
}

impl FnType {
    pub fn decl_fn(&self, decl: &fn(fnty: Type) -> ValueRef) -> ValueRef {
        let atys = self.arg_tys.iter().transform(|t| t.ty).collect::<~[Type]>();
        let rty = self.ret_ty.ty;
        let fnty = Type::func(atys, &rty);
        let llfn = decl(fnty);

        foreach (i, a) in self.attrs.iter().enumerate() {
            match *a {
                option::Some(attr) => {
                    unsafe {
                        let llarg = get_param(llfn, i);
                        llvm::LLVMAddAttribute(llarg, attr as c_uint);
                    }
                }
                _ => ()
            }
        }
        return llfn;
    }

    pub fn build_shim_args(&self, bcx: @mut Block, arg_tys: &[Type], llargbundle: ValueRef)
                           -> ~[ValueRef] {
        let mut atys: &[LLVMType] = self.arg_tys;
        let mut attrs: &[option::Option<Attribute>] = self.attrs;

        let mut llargvals = ~[];
        let mut i = 0u;
        let n = arg_tys.len();

        if self.sret {
            let llretptr = GEPi(bcx, llargbundle, [0u, n]);
            let llretloc = Load(bcx, llretptr);
                llargvals = ~[llretloc];
                atys = atys.tail();
                attrs = attrs.tail();
        }

        while i < n {
            let llargval = if atys[i].cast {
                let arg_ptr = GEPi(bcx, llargbundle, [0u, i]);
                let arg_ptr = BitCast(bcx, arg_ptr, atys[i].ty.ptr_to());
                Load(bcx, arg_ptr)
            } else if attrs[i].is_some() {
                GEPi(bcx, llargbundle, [0u, i])
            } else {
                load_inbounds(bcx, llargbundle, [0u, i])
            };
            llargvals.push(llargval);
            i += 1u;
        }

        return llargvals;
    }

    pub fn build_shim_ret(&self, bcx: @mut Block, arg_tys: &[Type], ret_def: bool,
                          llargbundle: ValueRef, llretval: ValueRef) {
        foreach (i, a) in self.attrs.iter().enumerate() {
            match *a {
                option::Some(attr) => {
                    unsafe {
                        llvm::LLVMAddInstrAttribute(llretval, (i + 1u) as c_uint, attr as c_uint);
                    }
                }
                _ => ()
            }
        }
        if self.sret || !ret_def {
            return;
        }
        let n = arg_tys.len();
        // R** llretptr = &args->r;
        let llretptr = GEPi(bcx, llargbundle, [0u, n]);
        // R* llretloc = *llretptr; /* (args->r) */
        let llretloc = Load(bcx, llretptr);
        if self.ret_ty.cast {
            let tmp_ptr = BitCast(bcx, llretloc, self.ret_ty.ty.ptr_to());
            // *args->r = r;
            Store(bcx, llretval, tmp_ptr);
        } else {
            // *args->r = r;
            Store(bcx, llretval, llretloc);
        };
    }

    pub fn build_wrap_args(&self, bcx: @mut Block, ret_ty: Type,
                           llwrapfn: ValueRef, llargbundle: ValueRef) {
        let mut atys: &[LLVMType] = self.arg_tys;
        let mut attrs: &[option::Option<Attribute>] = self.attrs;
        let mut j = 0u;
        let llretptr = if self.sret {
            atys = atys.tail();
            attrs = attrs.tail();
            j = 1u;
            get_param(llwrapfn, 0u)
        } else if self.ret_ty.cast {
            let retptr = alloca(bcx, self.ret_ty.ty, "");
            BitCast(bcx, retptr, ret_ty.ptr_to())
        } else {
            alloca(bcx, ret_ty, "")
        };

        let mut i = 0u;
        let n = atys.len();
        while i < n {
            let mut argval = get_param(llwrapfn, i + j);
            if attrs[i].is_some() {
                argval = Load(bcx, argval);
                store_inbounds(bcx, argval, llargbundle, [0u, i]);
            } else if atys[i].cast {
                let argptr = GEPi(bcx, llargbundle, [0u, i]);
                let argptr = BitCast(bcx, argptr, atys[i].ty.ptr_to());
                Store(bcx, argval, argptr);
            } else {
                store_inbounds(bcx, argval, llargbundle, [0u, i]);
            }
            i += 1u;
        }
        store_inbounds(bcx, llretptr, llargbundle, [0u, n]);
    }

    pub fn build_wrap_ret(&self, bcx: @mut Block, arg_tys: &[Type], llargbundle: ValueRef) {
        if self.ret_ty.ty.kind() == Void {
            return;
        }

        if bcx.fcx.llretptr.is_some() {
            let llretval = load_inbounds(bcx, llargbundle, [ 0, arg_tys.len() ]);
            let llretval = if self.ret_ty.cast {
                let retptr = BitCast(bcx, llretval, self.ret_ty.ty.ptr_to());
                Load(bcx, retptr)
            } else {
                Load(bcx, llretval)
            };
            let llretptr = BitCast(bcx, bcx.fcx.llretptr.get(), self.ret_ty.ty.ptr_to());
            Store(bcx, llretval, llretptr);
        }
    }
}
