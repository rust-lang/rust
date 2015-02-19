// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_upper_case_globals)]

use llvm;
use llvm::{TypeRef, Bool, False, True, TypeKind, ValueRef};
use llvm::{Float, Double, X86_FP80, PPC_FP128, FP128};

use trans::context::CrateContext;
use util::nodemap::FnvHashMap;

use syntax::ast;

use std::ffi::CString;
use std::mem;
use std::ptr;
use std::cell::RefCell;
use std::iter::repeat;

use libc::c_uint;

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(C)]
pub struct Type {
    rf: TypeRef
}

macro_rules! ty {
    ($e:expr) => ( Type::from_ref(unsafe { $e }))
}

/// Wrapper for LLVM TypeRef
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

    pub fn void(ccx: &CrateContext) -> Type {
        ty!(llvm::LLVMVoidTypeInContext(ccx.llcx()))
    }

    pub fn nil(ccx: &CrateContext) -> Type {
        Type::empty_struct(ccx)
    }

    pub fn metadata(ccx: &CrateContext) -> Type {
        ty!(llvm::LLVMMetadataTypeInContext(ccx.llcx()))
    }

    pub fn i1(ccx: &CrateContext) -> Type {
        ty!(llvm::LLVMInt1TypeInContext(ccx.llcx()))
    }

    pub fn i8(ccx: &CrateContext) -> Type {
        ty!(llvm::LLVMInt8TypeInContext(ccx.llcx()))
    }

    pub fn i16(ccx: &CrateContext) -> Type {
        ty!(llvm::LLVMInt16TypeInContext(ccx.llcx()))
    }

    pub fn i32(ccx: &CrateContext) -> Type {
        ty!(llvm::LLVMInt32TypeInContext(ccx.llcx()))
    }

    pub fn i64(ccx: &CrateContext) -> Type {
        ty!(llvm::LLVMInt64TypeInContext(ccx.llcx()))
    }

    // Creates an integer type with the given number of bits, e.g. i24
    pub fn ix(ccx: &CrateContext, num_bits: u64) -> Type {
        ty!(llvm::LLVMIntTypeInContext(ccx.llcx(), num_bits as c_uint))
    }

    pub fn f32(ccx: &CrateContext) -> Type {
        ty!(llvm::LLVMFloatTypeInContext(ccx.llcx()))
    }

    pub fn f64(ccx: &CrateContext) -> Type {
        ty!(llvm::LLVMDoubleTypeInContext(ccx.llcx()))
    }

    pub fn bool(ccx: &CrateContext) -> Type {
        Type::i8(ccx)
    }

    pub fn char(ccx: &CrateContext) -> Type {
        Type::i32(ccx)
    }

    pub fn i8p(ccx: &CrateContext) -> Type {
        Type::i8(ccx).ptr_to()
    }

    pub fn int(ccx: &CrateContext) -> Type {
        match &ccx.tcx().sess.target.target.target_pointer_width[] {
            "32" => Type::i32(ccx),
            "64" => Type::i64(ccx),
            tws => panic!("Unsupported target word size for int: {}", tws),
        }
    }

    pub fn int_from_ty(ccx: &CrateContext, t: ast::IntTy) -> Type {
        match t {
            ast::TyIs(_) => ccx.int_type(),
            ast::TyI8 => Type::i8(ccx),
            ast::TyI16 => Type::i16(ccx),
            ast::TyI32 => Type::i32(ccx),
            ast::TyI64 => Type::i64(ccx)
        }
    }

    pub fn uint_from_ty(ccx: &CrateContext, t: ast::UintTy) -> Type {
        match t {
            ast::TyUs(_) => ccx.int_type(),
            ast::TyU8 => Type::i8(ccx),
            ast::TyU16 => Type::i16(ccx),
            ast::TyU32 => Type::i32(ccx),
            ast::TyU64 => Type::i64(ccx)
        }
    }

    pub fn float_from_ty(ccx: &CrateContext, t: ast::FloatTy) -> Type {
        match t {
            ast::TyF32 => Type::f32(ccx),
            ast::TyF64 => Type::f64(ccx),
        }
    }

    pub fn func(args: &[Type], ret: &Type) -> Type {
        let vec : &[TypeRef] = unsafe { mem::transmute(args) };
        ty!(llvm::LLVMFunctionType(ret.to_ref(), vec.as_ptr(),
                                   args.len() as c_uint, False))
    }

    pub fn variadic_func(args: &[Type], ret: &Type) -> Type {
        let vec : &[TypeRef] = unsafe { mem::transmute(args) };
        ty!(llvm::LLVMFunctionType(ret.to_ref(), vec.as_ptr(),
                                   args.len() as c_uint, True))
    }

    pub fn struct_(ccx: &CrateContext, els: &[Type], packed: bool) -> Type {
        let els : &[TypeRef] = unsafe { mem::transmute(els) };
        ty!(llvm::LLVMStructTypeInContext(ccx.llcx(), els.as_ptr(),
                                          els.len() as c_uint,
                                          packed as Bool))
    }

    pub fn named_struct(ccx: &CrateContext, name: &str) -> Type {
        let name = CString::new(name).unwrap();
        ty!(llvm::LLVMStructCreateNamed(ccx.llcx(), name.as_ptr()))
    }

    pub fn empty_struct(ccx: &CrateContext) -> Type {
        Type::struct_(ccx, &[], false)
    }

    pub fn vtable(ccx: &CrateContext) -> Type {
        Type::array(&Type::i8p(ccx).ptr_to(), 1)
    }

    pub fn generic_glue_fn(cx: &CrateContext) -> Type {
        match cx.tn().find_type("glue_fn") {
            Some(ty) => return ty,
            None => ()
        }

        let ty = Type::glue_fn(cx, Type::i8p(cx));
        cx.tn().associate_type("glue_fn", &ty);

        ty
    }

    pub fn glue_fn(ccx: &CrateContext, t: Type) -> Type {
        Type::func(&[t], &Type::void(ccx))
    }

    pub fn tydesc(ccx: &CrateContext, str_slice_ty: Type) -> Type {
        let mut tydesc = Type::named_struct(ccx, "tydesc");
        let glue_fn_ty = Type::glue_fn(ccx, Type::i8p(ccx)).ptr_to();

        let int_ty = Type::int(ccx);

        // Must mirror:
        //
        // std::unstable::intrinsics::TyDesc

        let elems = [int_ty,     // size
                     int_ty,     // align
                     glue_fn_ty, // drop
                     str_slice_ty]; // name
        tydesc.set_struct_body(&elems, false);

        tydesc
    }

    pub fn array(ty: &Type, len: u64) -> Type {
        ty!(llvm::LLVMRustArrayType(ty.to_ref(), len))
    }

    pub fn vector(ty: &Type, len: u64) -> Type {
        ty!(llvm::LLVMVectorType(ty.to_ref(), len as c_uint))
    }

    pub fn vec(ccx: &CrateContext, ty: &Type) -> Type {
        Type::struct_(ccx,
            &[Type::array(ty, 0), Type::int(ccx)],
        false)
    }

    pub fn opaque_vec(ccx: &CrateContext) -> Type {
        Type::vec(ccx, &Type::i8(ccx))
    }

    pub fn vtable_ptr(ccx: &CrateContext) -> Type {
        Type::glue_fn(ccx, Type::i8p(ccx)).ptr_to().ptr_to()
    }

    pub fn opaque_trait(ccx: &CrateContext) -> Type {
        Type::struct_(ccx, &[Type::opaque_trait_data(ccx).ptr_to(), Type::vtable_ptr(ccx)], false)
    }

    pub fn opaque_trait_data(ccx: &CrateContext) -> Type {
        Type::i8(ccx)
    }

    pub fn kind(&self) -> TypeKind {
        unsafe {
            llvm::LLVMGetTypeKind(self.to_ref())
        }
    }

    pub fn set_struct_body(&mut self, els: &[Type], packed: bool) {
        unsafe {
            let vec : &[TypeRef] = mem::transmute(els);
            llvm::LLVMStructSetBody(self.to_ref(), vec.as_ptr(),
                                    els.len() as c_uint, packed as Bool)
        }
    }

    pub fn ptr_to(&self) -> Type {
        ty!(llvm::LLVMPointerType(self.to_ref(), 0))
    }

    pub fn is_aggregate(&self) -> bool {
        match self.kind() {
            TypeKind::Struct | TypeKind::Array => true,
            _ =>  false
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

    /// Return the number of elements in `self` if it is a LLVM vector type.
    pub fn vector_length(&self) -> uint {
        unsafe {
            llvm::LLVMGetVectorSize(self.to_ref()) as uint
        }
    }

    pub fn array_length(&self) -> uint {
        unsafe {
            llvm::LLVMGetArrayLength(self.to_ref()) as uint
        }
    }

    pub fn field_types(&self) -> Vec<Type> {
        unsafe {
            let n_elts = llvm::LLVMCountStructElementTypes(self.to_ref()) as uint;
            if n_elts == 0 {
                return Vec::new();
            }
            let mut elts: Vec<_> = repeat(Type { rf: ptr::null_mut() }).take(n_elts).collect();
            llvm::LLVMGetStructElementTypes(self.to_ref(),
                                            elts.as_mut_ptr() as *mut TypeRef);
            elts
        }
    }

    pub fn return_type(&self) -> Type {
        ty!(llvm::LLVMGetReturnType(self.to_ref()))
    }

    pub fn func_params(&self) -> Vec<Type> {
        unsafe {
            let n_args = llvm::LLVMCountParamTypes(self.to_ref()) as uint;
            let mut args: Vec<_> = repeat(Type { rf: ptr::null_mut() }).take(n_args).collect();
            llvm::LLVMGetParamTypes(self.to_ref(),
                                    args.as_mut_ptr() as *mut TypeRef);
            args
        }
    }

    pub fn float_width(&self) -> uint {
        match self.kind() {
            Float => 32,
            Double => 64,
            X86_FP80 => 80,
            FP128 | PPC_FP128 => 128,
            _ => panic!("llvm_float_width called on a non-float type")
        }
    }

    /// Retrieve the bit width of the integer type `self`.
    pub fn int_width(&self) -> u64 {
        unsafe {
            llvm::LLVMGetIntTypeWidth(self.to_ref()) as u64
        }
    }
}


/* Memory-managed object interface to type handles. */

pub struct TypeNames {
    named_types: RefCell<FnvHashMap<String, TypeRef>>,
}

impl TypeNames {
    pub fn new() -> TypeNames {
        TypeNames {
            named_types: RefCell::new(FnvHashMap())
        }
    }

    pub fn associate_type(&self, s: &str, t: &Type) {
        assert!(self.named_types.borrow_mut().insert(s.to_string(),
                                                     t.to_ref()).is_none());
    }

    pub fn find_type(&self, s: &str) -> Option<Type> {
        self.named_types.borrow().get(s).map(|x| Type::from_ref(*x))
    }

    pub fn type_to_string(&self, ty: Type) -> String {
        llvm::build_string(|s| unsafe {
                llvm::LLVMWriteTypeToString(ty.to_ref(), s);
            }).expect("non-UTF8 type description from LLVM")
    }

    pub fn types_to_str(&self, tys: &[Type]) -> String {
        let strs: Vec<String> = tys.iter().map(|t| self.type_to_string(*t)).collect();
        format!("[{}]", strs.connect(","))
    }

    pub fn val_to_string(&self, val: ValueRef) -> String {
        llvm::build_string(|s| unsafe {
                llvm::LLVMWriteValueToString(val, s);
            }).expect("nun-UTF8 value description from LLVM")
    }
}
