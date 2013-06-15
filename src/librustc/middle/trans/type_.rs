// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use lib::llvm::{llvm, TypeRef, Bool, False, True};

use middle::trans::context::CrateContext;
use middle::trans::base;

use syntax::{ast,abi};
use syntax::abi::{X86, X86_64, Arm, Mips};

use core::vec;

use core::libc::{c_uint};

pub struct Type {
    priv rf: TypeRef
}

macro_rules! ty (
    ($e:expr) => ( Type::from_ref(unsafe {

    }))
)

/**
 * Wrapper for LLVM TypeRef
 */
impl Type {
    pub fn from_ref(r: TypeRef) -> Type {
        Type {
            rf: r
        }
    }

    pub fn to_ref(&self) -> TypeRef {
        self.rf
    }

    pub fn void() -> Type {
        ty!(llvm::LLVMVoidTypeInContext(base::task_llcx()))
    }

    pub fn nil() -> Type {
        Type::empty_struct()
    }

    pub fn metadata() -> Type {
        ty!(llvm::LLVMMetadataTypeInContext(base::task_llcx()))
    }

    pub fn i1() -> Type {
        ty!(llvm::LLVMInt1TypeInContext(base::task_llcx()))
    }

    pub fn i8() -> Type {
        ty!(llvm::LLVMInt8TypeInContext(base::task_llcx()))
    }

    pub fn i16() -> Type {
        ty!(llvm::LLVMInt16TypeInContext(base::task_llcx()))
    }

    pub fn i32() -> Type {
        ty!(llvm::LLVMInt32TypeInContext(base::task_llcx()))
    }

    pub fn i64() -> Type {
        ty!(llvm::LLVMInt64TypeInContext(base::task_llcx()))
    }

    pub fn f32() -> Type {
        ty!(llvm::LLVMFloatTypeInContext(base::task_llcx()))
    }

    pub fn f64() -> Type {
        ty!(llvm::LLVMDoubleTypeInContext(base::task_llcx()))
    }

    pub fn bool() -> Type {
        Type::i8()
    }

    pub fn char() -> Type {
        Type::i32()
    }

    pub fn int(arch: abi::Architecture) -> Type {
        match arch {
            X86 | Arm | Mips => Type::i32(),
            X86_64 => Type::i64()
        }
    }

    pub fn float(_: abi::Architecture) -> Type {
        // All architectures currently just use doubles as the default
        // float size
        Type::f64()
    }

    pub fn int_from_ty(ctx: &CrateContext, t: ast::int_ty) -> Type {
        match t {
            ast::ty_i => ctx.int_type,
            ast::ty_char => Type::char(),
            ast::ty_i8 => Type::i8(),
            ast::ty_i16 => Type::i16(),
            ast::ty_i32 => Type::i32(),
            ast::ty_i64 => Type::i64()
        }
    }

    pub fn uint_from_ty(ctx: &CrateContext, t: ast::uint_ty) -> Type {
        match t {
            ast::ty_u => ctx.int_type,
            ast::ty_u8 => Type::i8(),
            ast::ty_u16 => Type::i16(),
            ast::ty_u32 => Type::i32(),
            ast::ty_u64 => Type::i64()
        }
    }

    pub fn float_from_ty(ctx: &CrateContext, t: ast::float_ty) -> Type {
        match t {
            ast::ty_f => ctx.float_ty,
            ast::ty_f32 => Type::f32(),
            ast::ty_f64 => Type::f64()
        }
    }

    pub fn size_t(arch: abi::Architecture) -> Type {
        Type::int(arch)
    }

    pub fn func(args: &[Type], ret: &Type) -> Type {
        let vec : &[TypeRef] = unsafe { cast::transmute() };
        ty!(llvm::LLVMFunctionType(ret.to_ref(), vec::raw::to_ptr(vec),
                                   args.len() as c_uint, False))
    }

    pub fn func_pair(cx: &CrateContext, fn_ty: &Type) -> Type {
        assert!(fn_ty.is_func(), "`fn_ty` must be a function type");

        Type::struct_([fn_ty.ptr_to(), Type::opaque_cbox_ptr(cx)], false)
    }

    pub fn ptr(ty: Type) -> Type {
        ty!(llvm::LLVMPointerType(ty, 0 as c_uint))
    }

    pub fn struct_(els: &[Type], packed: bool) -> Type {
        let els : &[TypeRef] = unsafe { cast::transmute(els) };
        ty!(llvm::LLVMStructType(vec::raw::to_ptr(els), els.len() as c_uint, packed as Bool))
    }

    pub fn named_struct(name: &str) -> Type {
        let ctx = base::task_llcx();
        ty!(name.as_c_str(|s| llvm::LLVMStructCreateNamed(ctx, s)))
    }

    pub fn empty_struct() -> Type {
        Type::struct_([], false)
    }

    pub fn vtable() -> Type {
        Type::array(Type::i8().ptr_to(), 1)
    }

    pub fn generic_glue_fn(cx: &mut CrateContext) -> Type {
        match cx.tn.find_type("glue_fn") {
            Some(ty) => return ty,
            None => ()
        }

        let ty = cx.tydesc_type.get_field(abi::tydesc_field_drop_glue);
        cx.tn.associate_type("glue_fn", ty);

        return ty;
    }

    pub fn tydesc(arch: abi::Architecture) -> Type {
        let mut tydesc = Type::named_struct("tydesc");
        let tydescpp = tydesc.ptr_to().ptr_to();
        let pvoid = Type::i8().ptr_to();
        let glue_fn_ty = Type::func(
            [ Type::nil.ptr_to(), tydescpp, pvoid ],
            Type::void()).ptr_to();

        let int_ty = Type::int(arch);

        let elems = [
            int_type, int_type,
            glue_fn_ty, glue_fn_ty, glue_fn_ty, glue_fn_ty,
            pvoid, pvoid
        ];

        tydesc.set_struct_body(elems, false);

        return tydesc;
    }

    pub fn array(ty: &Type, len: uint) -> Type {
        ty!(llvm::LLVMArrayType(ty.to_ref(), len as c_uint))
    }

    pub fn vector(ty: &Type, len: uint) -> Type {
        ty!(llvm::LLVMVectorType(ty.to_ref(), len as c_uint))
    }

    pub fn vec(arch: abi::Architecture, ty: &Type) -> Type {
        Type::struct_(
            [ Type::int(arch), Type::int(arch), Type::array(ty, 0) ],
        false)
    }

    pub fn opaque_vec(arch: abi::Architecture) -> Type {
        Type::vec(arch, Type::i8())
    }

    #[inline]
    pub fn box_header_fields(ctx: &CrateContext) -> ~[Type] {
        ~[
            ctx.int_type, ctx.tydesc_type.ptr_to(),
            Type::i8().ptr_to(), Type::i8().ptr_to()
        ]
    }

    pub fn box_header(ctx: &CrateContext) -> Type {
        Type::struct_(Type::box_header_fields(ctx), false)
    }

    pub fn box(ctx: &CrateContext, ty: &Type) -> Type {
        Type::struct_(Type::box_header_fields(ctx) + [t], false)
    }

    pub fn opaque_box(ctx: &CrateContext) -> Type {
        Type::box(ctx, Type::i8())
    }

    pub fn unique(ctx: &CrateContext, ty: &Type) -> Type {
        Type::box(ctx, ty)
    }

    pub fn opaque_cbox_ptr(cx: &CrateContext) -> Type {
        Type::opaque_box().ptr_to()
    }

    pub fn enum_discrim(cx: &CrateContext) -> Type {
        cx.int_type
    }

    pub fn set_struct_body(&mut self, els: &[Type], packed: bool) {
        assert!(self.is_struct(), "Type must be a struct");

        unsafe {
            let vec : &[TypeRef] = cast::transmute(els);
            llvm::LLVMStructSetBody(self.to_ref(), to_ptr(vec),
                                    els.len() as c_uint, packed as Bool)
        }
    }
}
