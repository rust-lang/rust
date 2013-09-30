// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use lib::llvm::{llvm, ValueRef, TypeRef, Bool, False, True};
use lib::llvm::{ContextRef, TypeKind, TypeNames};
use lib::llvm::{Float, Double, X86_FP80, PPC_FP128, FP128};

use middle::ty;

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

    pub fn func(args: &[Type], ret: &Type) -> Type {
        let vec : &[TypeRef] = unsafe { cast::transmute(args) };
        ty!(llvm::LLVMFunctionType(ret.to_ref(), vec::raw::to_ptr(vec),
                                   args.len() as c_uint, False))
    }

    pub fn ptr(ty: Type) -> Type {
        ty!(llvm::LLVMPointerType(ty.to_ref(), 0 as c_uint))
    }

    pub fn kind(&self) -> TypeKind {
        unsafe {
            llvm::LLVMGetTypeKind(self.to_ref())
        }
    }

    pub fn set_struct_body(&mut self, els: &[Type], packed: bool) {
        unsafe {
            let vec : &[TypeRef] = cast::transmute(els);
            llvm::LLVMStructSetBody(self.to_ref(), vec::raw::to_ptr(vec),
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

            llvm::LLVMGetStructElementTypes(self.to_ref(), vec::raw::to_mut_ptr(elems));

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
            llvm::LLVMGetParamTypes(self.to_ref(), vec::raw::to_ptr(args));
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

/**
 * Crate type set
 */

pub struct CrateTypes {
    priv llcx: ContextRef,
    priv tn: TypeNames,
    priv i1_t: Type,
    priv i8_t: Type,
    priv i16_t: Type,
    priv i32_t: Type,
    priv i64_t: Type,
    priv f32_t: Type,
    priv f64_t: Type,
    priv i8p_t: Type,
    priv tydesc_t: Type,
    priv int_t: Type,
    priv float_t: Type,
    priv nil_t: Type,
    priv void_t: Type,
    priv str_slice_t: Type,
    priv generic_glue_fn_t: Type,
    priv opaque_box_t: Type,
    priv box_header_fields: ~[Type],
    priv opaque_vec_t: Type
}

impl CrateTypes {

    /**
     * Constriction/initialization
     */
    pub fn new(arch: Architecture, llcx: ContextRef) -> @CrateTypes {
        let mut tn = TypeNames::new();

        // init primitive types
        let i1_type = ty!(llvm::LLVMInt1TypeInContext(llcx));
        let i8_type = ty!(llvm::LLVMInt8TypeInContext(llcx));
        let i16_type = ty!(llvm::LLVMInt16TypeInContext(llcx));
        let i32_type = ty!(llvm::LLVMInt32TypeInContext(llcx));
        let i64_type = ty!(llvm::LLVMInt64TypeInContext(llcx));
        let f32_type = ty!(llvm::LLVMFloatTypeInContext(llcx));
        let f64_type = ty!(llvm::LLVMDoubleTypeInContext(llcx));
        let i8p_type = i8_type.ptr_to();
        let int_type = match arch {
            X86 | Arm | Mips => i32_type,
            X86_64 => i64_type
        };
        let float_type = f64_type; // All architecture float are defaulted to doubles
        let nil_type = ty!(llvm::LLVMStructTypeInContext(llcx, vec::raw::to_ptr(&[]),
                                                         0, False));
        let void_type = ty!(llvm::LLVMVoidTypeInContext(llcx));

        // init str_slice
        let mut str_slice_ty =
            ty!("str_slice".with_c_str(|s| llvm::LLVMStructCreateNamed(llcx, s)));
        str_slice_ty.set_struct_body([i8p_type, int_type], false);
        tn.associate_type("str_slice", &str_slice_ty);

        // init glue_fn type
        let glue_fn_ty = CrateTypes::func_([nil_type.ptr_to(), i8p_type], &void_type);
        tn.associate_type("glue_fn", &glue_fn_ty);

        // init tydesc
        let mut tydesc_type =  ty!("tydesc".with_c_str(|s| llvm::LLVMStructCreateNamed(llcx, s)));
        let glue_fn_ptr = glue_fn_ty.ptr_to();
        // Must mirror: std::unstable::intrinsics::TyDesc AND type_desc in rt
        let elems = [int_type,    // size
                     int_type,    // align
                     glue_fn_ptr, // take
                     glue_fn_ptr, // drop
                     glue_fn_ptr, // free
                     glue_fn_ptr, // visit
                     int_type,    // borrow_offset
                     ty!(llvm::LLVMStructTypeInContext(llcx,  // name
                                                       vec::raw::to_ptr([i8p_type.to_ref(),
                                                                         int_type.to_ref()]),
                                                       2, False))];
        tydesc_type.set_struct_body(elems, false);
        tn.associate_type("tydesc", &tydesc_type);

        // init opaque_vec
        let mut opaque_vec_type =
            ty!("opaque_vec".with_c_str(|s| llvm::LLVMStructCreateNamed(llcx, s)));
        opaque_vec_type.set_struct_body([int_type, int_type,
                                         ty!(llvm::LLVMArrayType(i8_type.to_ref(), 0))],
                                        false);
        tn.associate_type("opaque_vec", &opaque_vec_type);

        // init opaque_box
        let box_h_fields = ~[int_type, tydesc_type.ptr_to(), i8p_type, i8p_type];
        let obox_h_fields = [int_type.to_ref(),
                             tydesc_type.ptr_to().to_ref(),
                             i8p_type.to_ref(),
                             i8p_type.to_ref(),
                             i8_type.to_ref()];
        let opaque_box_type = ty!(llvm::LLVMStructTypeInContext(
            llcx, vec::raw::to_ptr(obox_h_fields), obox_h_fields.len() as c_uint, False));

        // form the result struct
        @CrateTypes {
            llcx: llcx,
            tn: tn,
            i1_t: i1_type,
            i8_t: i8_type,
            i16_t: i16_type,
            i32_t: i32_type,
            i64_t: i64_type,
            f32_t: f32_type,
            f64_t: f64_type,
            i8p_t: i8p_type,
            nil_t: nil_type,
            void_t: void_type,
            tydesc_t: tydesc_type,
            int_t: int_type,
            float_t: float_type,
            str_slice_t: str_slice_ty,
            generic_glue_fn_t: glue_fn_ty,
            opaque_box_t: opaque_box_type,
            box_header_fields: box_h_fields,
            opaque_vec_t: opaque_vec_type
        }
    }

    /**
     * Primitive type getters
     */

    #[inline(always)]
    pub fn i(&self) -> Type {
        self.int_t
    }

    #[inline(always)]
    pub fn f(&self) -> Type {
        self.float_t
    }

    #[inline(always)]
    pub fn void(&self) -> Type {
        self.void_t
    }

    #[inline(always)]
    pub fn nil(&self) -> Type {
        self.nil_t
    }

    pub fn metadata(&self) -> Type {
        ty!(llvm::LLVMMetadataTypeInContext(self.llcx))
    }

    pub fn i1(&self) -> Type {
        self.i1_t
    }

    #[inline(always)]
    pub fn i8(&self) -> Type {
        self.i8_t
    }

    #[inline(always)]
    pub fn i16(&self) -> Type {
        self.i16_t
    }

    #[inline(always)]
    pub fn i32(&self) -> Type {
        self.i32_t
    }

    #[inline(always)]
    pub fn i64(&self) -> Type {
        self.i64_t
    }

    #[inline(always)]
    pub fn f32(&self) -> Type {
        self.f32_t
    }

    #[inline(always)]
    pub fn f64(&self) -> Type {
        self.f64_t
    }

    #[inline(always)]
    pub fn bool(&self) -> Type {
        self.i8()
    }

    #[inline(always)]
    pub fn char(&self) -> Type {
        self.i32()
    }

    #[inline(always)]
    pub fn i8p(&self) -> Type {
        self.i8p_t
    }

    pub fn int_by_size(&self, size: uint) -> Type {
        ty!(llvm::LLVMIntTypeInContext(self.llcx, size as c_uint))
    }


    pub fn int_from_ast_ty(&self, t: ast::int_ty) -> Type {
        match t {
            ast::ty_i => self.i(),
            ast::ty_i8 => self.i8(),
            ast::ty_i16 => self.i16(),
            ast::ty_i32 => self.i32(),
            ast::ty_i64 => self.i64()
        }
    }

    pub fn uint_from_ast_ty(&self, t: ast::uint_ty) -> Type {
        match t {
            ast::ty_u => self.i(),
            ast::ty_u8 => self.i8(),
            ast::ty_u16 => self.i16(),
            ast::ty_u32 => self.i32(),
            ast::ty_u64 => self.i64()
        }
    }

    pub fn float_from_ast_ty(&self, t: ast::float_ty) -> Type {
        match t {
            ast::ty_f => self.f(),
            ast::ty_f32 => self.f32(),
            ast::ty_f64 => self.f64()
        }
    }

    pub fn size_t(&self) -> Type {
        self.i()
    }

    /**
     * Complex type getters
     */

    #[inline(always)]
    pub fn tydesc(&self) -> Type {
        self.tydesc_t
    }

    #[inline(always)]
    pub fn opaque_vec(&self) -> Type {
        self.opaque_vec_t
    }

    #[inline(always)]
    pub fn str_slice(&self) -> Type {
        self.str_slice_t
    }

    pub fn func_(args: &[Type], ret: &Type) -> Type {
        let vec : &[TypeRef] = unsafe { cast::transmute(args) };
        ty!(llvm::LLVMFunctionType(ret.to_ref(), vec::raw::to_ptr(vec),
                                   args.len() as c_uint, False))
    }

    #[inline(always)]
    pub fn func(&self, args: &[Type], ret: &Type) -> Type {
        CrateTypes::func_(args, ret)
    }

    pub fn func_pair(&self, fn_ty: &Type) -> Type {
        self.struct_([fn_ty.ptr_to(), self.opaque_cbox_ptr()], false)
    }

    pub fn struct_(&self, els: &[Type], packed: bool) -> Type {
        let els : &[TypeRef] = unsafe { cast::transmute(els) };
        ty!(llvm::LLVMStructTypeInContext(self.llcx, vec::raw::to_ptr(els),
                                          els.len() as c_uint, packed as Bool))
    }

    pub fn named_struct(&self, name: &str) -> Type {
        ty!(name.with_c_str(|s| llvm::LLVMStructCreateNamed(self.llcx, s)))
    }

    pub fn empty_struct(&self) -> Type {
        self.struct_([], false)
    }

    pub fn vtable(&self) -> Type {
        self.array(&self.i8().ptr_to(), 1)
    }

    #[inline(always)]
    pub fn generic_glue_fn(&self) -> Type {
        self.generic_glue_fn_t
    }

    pub fn glue_fn(&self, t: Type) -> Type {
        self.func([ self.nil().ptr_to(), t ],
            &self.void())
    }

    pub fn array(&self, ty: &Type, len: u64) -> Type {
        ty!(llvm::LLVMArrayType(ty.to_ref(), len as c_uint))
    }

    pub fn vector(&self, ty: &Type, len: u64) -> Type {
        ty!(llvm::LLVMVectorType(ty.to_ref(), len as c_uint))
    }

    pub fn vec(&self, ty: &Type) -> Type {
        self.struct_(
            [ self.i(), self.i(), self.array(ty, 0) ],
        false)
    }

    pub fn box(&self, ty: &Type) -> Type {
        self.struct_(self.box_header_fields + &[*ty], false)
    }

    pub fn opaque(&self) -> Type {
        self.i8()
    }

    pub fn opaque_box(&self) -> Type {
        self.opaque_box_t
    }

    pub fn unique(&self, ty: &Type) -> Type {
        self.box(ty)
    }

    pub fn opaque_cbox_ptr(&self) -> Type {
        self.opaque_box().ptr_to()
    }

    pub fn enum_discrim(&self) -> Type {
        self.i()
    }

    pub fn opaque_trait(&self, store: ty::TraitStore) -> Type {
        let tydesc_ptr = self.tydesc().ptr_to();
        let box_ty = match store {
            ty::BoxTraitStore => self.opaque_box(),
            ty::UniqTraitStore => self.unique(&self.i8()),
            ty::RegionTraitStore(*) => self.i8()
        };
        self.struct_([tydesc_ptr, box_ty.ptr_to()], false)
    }


    /**
     * ValueRef packers
     */

     pub fn struct_val(&self, elts: &[ValueRef]) -> ValueRef {
         unsafe {
             do elts.as_imm_buf |ptr, len| {
                 llvm::LLVMConstStructInContext(self.llcx, ptr, len as c_uint, False)
             }
         }
     }

    /**
     * Utility functions
     */

    pub fn type_to_str(&self, ty: Type) -> ~str {
        self.tn.type_to_str(ty)
    }

    pub fn val_to_str(&self, val: ValueRef) -> ~str {
        self.tn.val_to_str(val)
    }

    pub fn types_to_str(&self, tys: &[Type]) -> ~str {
        self.tn.types_to_str(tys)
    }
}
