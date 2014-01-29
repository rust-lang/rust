// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(non_uppercase_pattern_statics)];

use lib::llvm::{llvm, TypeRef, Bool, False, True, TypeKind};
use lib::llvm::{Float, Double, X86_FP80, PPC_FP128, FP128};

use middle::ty;

use middle::trans::context::CrateContext;
use middle::trans::base;

use syntax::ast;
use syntax::abi::{Architecture, X86, X86_64, Arm, Mips};

use std::c_str::ToCStr;
use std::vec;
use std::cast;

use std::libc::{c_uint};

#[deriving(Clone, Eq)]
pub struct Type {
    priv rf: TypeRef
}

macro_rules! ty (
    ($e:expr) => ( Type::from_ref(unsafe { $e }))
)

/**
 * Wrapper for LLVM TypeRef
 */
impl Type {
    #[inline(always)]
    pub fn from_ref(r: TypeRef) -> Type {
        Type {
            rf: r
        }
    }

    #[inline(always)] // So it doesn't kill --opt-level=0 builds of the compiler
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

    pub fn i8p() -> Type {
        Type::i8().ptr_to()
    }

    pub fn int(arch: Architecture) -> Type {
        match arch {
            X86 | Arm | Mips => Type::i32(),
            X86_64 => Type::i64()
        }
    }

    pub fn float(_: Architecture) -> Type {
        // All architectures currently just use doubles as the default
        // float size
        Type::f64()
    }

    pub fn int_from_ty(ctx: &CrateContext, t: ast::IntTy) -> Type {
        match t {
            ast::TyI => ctx.int_type,
            ast::TyI8 => Type::i8(),
            ast::TyI16 => Type::i16(),
            ast::TyI32 => Type::i32(),
            ast::TyI64 => Type::i64()
        }
    }

    pub fn uint_from_ty(ctx: &CrateContext, t: ast::UintTy) -> Type {
        match t {
            ast::TyU => ctx.int_type,
            ast::TyU8 => Type::i8(),
            ast::TyU16 => Type::i16(),
            ast::TyU32 => Type::i32(),
            ast::TyU64 => Type::i64()
        }
    }

    pub fn float_from_ty(t: ast::FloatTy) -> Type {
        match t {
            ast::TyF32 => Type::f32(),
            ast::TyF64 => Type::f64()
        }
    }

    pub fn size_t(arch: Architecture) -> Type {
        Type::int(arch)
    }

    pub fn func(args: &[Type], ret: &Type) -> Type {
        let vec : &[TypeRef] = unsafe { cast::transmute(args) };
        ty!(llvm::LLVMFunctionType(ret.to_ref(), vec.as_ptr(),
                                   args.len() as c_uint, False))
    }

    pub fn variadic_func(args: &[Type], ret: &Type) -> Type {
        let vec : &[TypeRef] = unsafe { cast::transmute(args) };
        ty!(llvm::LLVMFunctionType(ret.to_ref(), vec.as_ptr(),
                                   args.len() as c_uint, True))
    }

    pub fn ptr(ty: Type) -> Type {
        ty!(llvm::LLVMPointerType(ty.to_ref(), 0 as c_uint))
    }

    pub fn struct_(els: &[Type], packed: bool) -> Type {
        let els : &[TypeRef] = unsafe { cast::transmute(els) };
        ty!(llvm::LLVMStructTypeInContext(base::task_llcx(), els.as_ptr(),
                                          els.len() as c_uint, packed as Bool))
    }

    pub fn named_struct(name: &str) -> Type {
        let ctx = base::task_llcx();
        ty!(name.with_c_str(|s| llvm::LLVMStructCreateNamed(ctx, s)))
    }

    pub fn empty_struct() -> Type {
        Type::struct_([], false)
    }

    pub fn vtable() -> Type {
        Type::array(&Type::i8p().ptr_to(), 1)
    }

    pub fn generic_glue_fn(cx: &CrateContext) -> Type {
        match cx.tn.find_type("glue_fn") {
            Some(ty) => return ty,
            None => ()
        }

        let ty = Type::glue_fn(Type::i8p());
        cx.tn.associate_type("glue_fn", &ty);

        return ty;
    }

    pub fn glue_fn(t: Type) -> Type {
        Type::func([t], &Type::void())
    }

    pub fn tydesc(arch: Architecture) -> Type {
        let mut tydesc = Type::named_struct("tydesc");
        let glue_fn_ty = Type::glue_fn(Type::i8p()).ptr_to();

        let int_ty = Type::int(arch);

        // Must mirror:
        //
        // std::unstable::intrinsics::TyDesc

        let elems = [int_ty,     // size
                     int_ty,     // align
                     glue_fn_ty, // drop
                     glue_fn_ty, // visit
                     Type::struct_([Type::i8p(), Type::int(arch)], false)]; // name
        tydesc.set_struct_body(elems, false);

        return tydesc;
    }

    pub fn array(ty: &Type, len: u64) -> Type {
        ty!(llvm::LLVMArrayType(ty.to_ref(), len as c_uint))
    }

    pub fn vector(ty: &Type, len: u64) -> Type {
        ty!(llvm::LLVMVectorType(ty.to_ref(), len as c_uint))
    }

    pub fn vec(arch: Architecture, ty: &Type) -> Type {
        Type::struct_(
            [ Type::int(arch), Type::int(arch), Type::array(ty, 0) ],
        false)
    }

    pub fn opaque_vec(arch: Architecture) -> Type {
        Type::vec(arch, &Type::i8())
    }

    // The box pointed to by @T.
    pub fn at_box(ctx: &CrateContext, ty: Type) -> Type {
        Type::struct_([
            ctx.int_type, ctx.tydesc_type.ptr_to(),
            Type::i8p(), Type::i8p(), ty
        ], false)
    }

    pub fn opaque_trait(ctx: &CrateContext, store: ty::TraitStore) -> Type {
        let vtable = Type::glue_fn(Type::i8p()).ptr_to().ptr_to();
        let box_ty = match store {
            ty::BoxTraitStore => Type::at_box(ctx, Type::i8()),
            ty::UniqTraitStore => Type::i8(),
            ty::RegionTraitStore(..) => Type::i8()
        };
        Type::struct_([vtable, box_ty.ptr_to()], false)
    }

    pub fn kind(&self) -> TypeKind {
        unsafe {
            llvm::LLVMGetTypeKind(self.to_ref())
        }
    }

    pub fn set_struct_body(&mut self, els: &[Type], packed: bool) {
        unsafe {
            let vec : &[TypeRef] = cast::transmute(els);
            llvm::LLVMStructSetBody(self.to_ref(), vec.as_ptr(),
                                    els.len() as c_uint, packed as Bool)
        }
    }

    pub fn ptr_to(&self) -> Type {
        ty!(llvm::LLVMPointerType(self.to_ref(), 0))
    }

    pub fn get_field(&self, idx: uint) -> Type {
        unsafe {
            let num_fields = llvm::LLVMCountStructElementTypes(self.to_ref()) as uint;
            let mut elems = vec::from_elem(num_fields, 0 as TypeRef);

            llvm::LLVMGetStructElementTypes(self.to_ref(), elems.as_mut_ptr());

            Type::from_ref(elems[idx])
        }
    }

    pub fn is_packed(&self) -> bool {
        unsafe {
            llvm::LLVMIsPackedStruct(self.to_ref()) == True
        }
    }

    pub fn element_type(&self) -> Type {
        unsafe {
            Type::from_ref(llvm::LLVMGetElementType(self.to_ref()))
        }
    }

    pub fn array_length(&self) -> uint {
        unsafe {
            llvm::LLVMGetArrayLength(self.to_ref()) as uint
        }
    }

    pub fn field_types(&self) -> ~[Type] {
        unsafe {
            let n_elts = llvm::LLVMCountStructElementTypes(self.to_ref()) as uint;
            if n_elts == 0 {
                return ~[];
            }
            let mut elts = vec::from_elem(n_elts, 0 as TypeRef);
            llvm::LLVMGetStructElementTypes(self.to_ref(), &mut elts[0]);
            cast::transmute(elts)
        }
    }

    pub fn return_type(&self) -> Type {
        ty!(llvm::LLVMGetReturnType(self.to_ref()))
    }

    pub fn func_params(&self) -> ~[Type] {
        unsafe {
            let n_args = llvm::LLVMCountParamTypes(self.to_ref()) as uint;
            let args = vec::from_elem(n_args, 0 as TypeRef);
            llvm::LLVMGetParamTypes(self.to_ref(), args.as_ptr());
            cast::transmute(args)
        }
    }

    pub fn float_width(&self) -> uint {
        match self.kind() {
            Float => 32,
            Double => 64,
            X86_FP80 => 80,
            FP128 | PPC_FP128 => 128,
            _ => fail!("llvm_float_width called on a non-float type")
        }
    }
}
