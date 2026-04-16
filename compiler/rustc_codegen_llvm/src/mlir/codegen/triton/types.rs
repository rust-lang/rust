/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use melior::Context;
use melior::ir::Type;
use melior::ir::r#type::{IntegerType, TupleType};
use rustc_ast::{FloatTy, IntTy, UintTy};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{
    AdtDef, AliasTy, AliasTyKind, GenericArg, ParamTy, Ty, TyCtxt, TyKind, TypingEnv,
};

/// Returns true if `ty` is `core::option::Option<_>` or any custom `Option<_>` ADT
/// (e.g. the `pub enum Option<T>` defined directly in `no_core` crates like
/// `triton_kitchen_sink`).
pub fn is_option_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    if let TyKind::Adt(adt_def, _) = ty.kind() {
        let adt_path = with_no_trimmed_paths!(tcx.def_path_str(adt_def.did()));
        adt_path == "core::option::Option"
            || adt_path == "std::option::Option"
            || adt_path == "Option"
            || adt_path.ends_with("::Option")
    } else {
        false
    }
}
use rustc_mlir::shared::builtin::tensor_type;
use rustc_mlir::triton::pointer_type;

type AdtHandler =
    for<'tcx, 'c> fn(&TypeMapper, &'c Context, &TyCtxt<'tcx>, &[GenericArg<'tcx>]) -> Type<'c>;

use std::collections::HashMap;
use std::sync::OnceLock;

static ADT_HANDLER_MAP: OnceLock<HashMap<&'static str, AdtHandler>> = OnceLock::new();

pub struct TypeMapper {}

impl TypeMapper {
    pub fn new() -> Self {
        Self {}
    }

    pub fn map_type<'tcx, 'c>(
        &self,
        context: &'c Context,
        tcx: &TyCtxt<'tcx>,
        ty: &Ty<'tcx>,
    ) -> Type<'c> {
        match ty.kind() {
            TyKind::Int(int_ty) => self.create_int_type(context, tcx, int_ty),
            TyKind::Uint(uint_ty) => self.create_uint_type(context, tcx, uint_ty),
            TyKind::Float(float_ty) => self.create_float_type(context, tcx, float_ty),
            TyKind::Bool => self.create_bool_type(context),
            TyKind::Array(_elem_ty, _len) => todo!("Array: {:?} {:?}", _elem_ty, _len),
            TyKind::Char => todo!("Char"),
            TyKind::Adt(def, args) => self.map_adt_ty(context, tcx, def, args.as_slice()),
            TyKind::Foreign(_id) => todo!("Foreign: {:?}", _id),
            TyKind::Str => todo!("Str"),
            TyKind::Pat(_ty, _pat) => todo!("Pat: {:?} {:?}", _ty, _pat),
            TyKind::Slice(_ty) => todo!("Slice: {:?}", _ty),
            TyKind::RawPtr(ty, _mutability) => self.create_raw_ptr_type(context, tcx, ty),
            TyKind::Ref(_region, inner_ty, _mutability) => {
                // A reference to a slice (&[T]) is a fat pointer (ptr + len).
                // We represent it as a pair of i64 values (opaque); the function body
                // of Triton intrinsics that take &[T] is never actually executed on GPU —
                // calls are intercepted at call-site by the codegen dispatch table.
                // For type-mapping purposes, return i64 as a stand-in.
                match inner_ty.kind() {
                    TyKind::Slice(_) => IntegerType::new(context, 64).into(),
                    _ => self.map_type(context, tcx, inner_ty),
                }
            }
            TyKind::FnDef(_def, _args) => todo!("FnDef: {:?} {:?}", _def, _args),
            TyKind::FnPtr(_binder, _fn_header) => todo!("FnPtr: {:?} {:?}", _binder, _fn_header),
            TyKind::UnsafeBinder(_unsafe_binder_inner) => {
                todo!("UnsafeBinder: {:?}", _unsafe_binder_inner)
            }
            TyKind::Dynamic(_existential_predicates, _region) => {
                todo!("Dynamic: {:?} {:?}", _existential_predicates, _region)
            }
            TyKind::Closure(_def, _args) => todo!("Closure: {:?} {:?}", _def, _args),
            TyKind::CoroutineClosure(_def, _args) => {
                todo!("CoroutineClosure: {:?} {:?}", _def, _args)
            }
            TyKind::Coroutine(_def, _args) => todo!("Coroutine: {:?} {:?}", _def, _args),
            TyKind::CoroutineWitness(_def, _args) => {
                todo!("CoroutineWitness: {:?} {:?}", _def, _args)
            }
            TyKind::Never => todo!("Never"),
            TyKind::Tuple(tys) => self.create_tuple_type(context, tcx, tys.as_slice()),
            TyKind::Alias(alias_ty_kind, alias_ty) => {
                self.map_alias_ty(context, tcx, ty, alias_ty_kind, alias_ty)
            }
            TyKind::Param(_param_ty) => self.create_param_type(context, tcx, _param_ty),
            TyKind::Bound(bound_var_index_kind, _bound_ty) => {
                todo!("Bound: {:?} {:?}", bound_var_index_kind, _bound_ty)
            }
            TyKind::Placeholder(_placeholder_ty) => todo!("Placeholder: {:?}", _placeholder_ty),
            TyKind::Infer(_infer_ty) => todo!("Infer: {:?}", _infer_ty),
            TyKind::Error(_error_guaranteed) => todo!("Error: {:?}", _error_guaranteed),
        }
    }

    fn map_adt_ty<'tcx, 'c>(
        &self,
        context: &'c Context,
        tcx: &TyCtxt<'tcx>,
        def: &AdtDef,
        args: &[GenericArg<'tcx>],
    ) -> Type<'c> {
        let name = with_no_trimmed_paths!(tcx.def_path_str(def.did()));
        println!("map_adt_ty: name:{:?} {:?} {:?}", name, def, args);

        // Try the registered handler first; fall back to i32 for fieldless enums.
        let map = ADT_HANDLER_MAP.get_or_init(|| {
            let entries: Vec<(&'static str, AdtHandler)> = vec![
                ("triton::llvm::triton::tensor::Tensor", triton_tensor_handler),
                ("triton::llvm::triton::pointer::Pointer", triton_pointer_handler),
                ("triton::llvm::triton::types::Bool", triton_bool_handler),
                ("triton::Axis", triton_program_axis_handler),
            ];
            entries.into_iter().collect()
        });

        if let Some(handler) = map.get(name.as_str()) {
            handler(self, context, tcx, args)
        } else if def.is_enum() {
            // Fieldless enums (DotFormat, InputPrecision, FpDowncastRounding, etc.)
            // are represented as their discriminant integer, which is i32.
            IntegerType::new(context, 32).into()
        } else {
            panic!("Handler not found for ADT: {:?}", name)
        }
    }

    fn map_alias_ty<'tcx, 'c>(
        &self,
        context: &'c Context,
        tcx: &TyCtxt<'tcx>,
        ty: &Ty<'tcx>,
        _alias_ty_kind: &AliasTyKind,
        alias_ty: &AliasTy<'tcx>,
    ) -> Type<'c> {
        let typing_env = TypingEnv::post_analysis(*tcx, alias_ty.def_id);
        let normalized = tcx.try_normalize_erasing_regions(typing_env, *ty);
        if let Ok(normalized) = normalized {
            self.map_type(context, tcx, &normalized)
        } else {
            panic!("Could not normalize Alias: {:?} {:?}", ty, alias_ty);
        }
    }

    fn create_param_type<'tcx, 'c>(
        &self,
        _context: &'c Context,
        _tcx: &TyCtxt<'tcx>,
        param_ty: &ParamTy,
    ) -> Type<'c> {
        todo!("Param: {:?}", param_ty);
    }

    fn create_int_type<'tcx, 'c>(
        &self,
        context: &'c Context,
        _tcx: &TyCtxt<'tcx>,
        int_ty: &IntTy,
    ) -> Type<'c> {
        let num_bits = match int_ty {
            IntTy::Isize => unimplemented!("isize is not supported as it is device-dependent"),
            IntTy::I8 => 8,
            IntTy::I16 => 16,
            IntTy::I32 => 32,
            IntTy::I64 => 64,
            IntTy::I128 => 128,
        };

        IntegerType::new(context, num_bits).into()
    }

    fn create_uint_type<'tcx, 'c>(
        &self,
        context: &'c Context,
        _tcx: &TyCtxt<'tcx>,
        uint_ty: &UintTy,
    ) -> Type<'c> {
        let num_bits = match uint_ty {
            UintTy::Usize => unimplemented!("usize is not supported as it is device-dependent"),
            UintTy::U8 => 8,
            UintTy::U16 => 16,
            UintTy::U32 => 32,
            UintTy::U64 => 64,
            UintTy::U128 => 128,
        };

        // for the moment we use the signless variant of the integer type
        IntegerType::new(context, num_bits).into()
    }

    fn create_float_type<'tcx, 'c>(
        &self,
        context: &'c Context,
        _tcx: &TyCtxt<'tcx>,
        float_ty: &FloatTy,
    ) -> Type<'c> {
        match float_ty {
            FloatTy::F16 => Type::float16(context),
            FloatTy::F32 => Type::float32(context),
            FloatTy::F64 => Type::float64(context),
            FloatTy::F128 => unimplemented!("f128 is not supported"),
        }
    }

    fn create_bool_type<'c>(&self, context: &'c Context) -> Type<'c> {
        // bools are 1-bit integers
        IntegerType::new(context, 1).into()
    }

    fn create_tuple_type<'tcx, 'c>(
        &self,
        context: &'c Context,
        tcx: &TyCtxt<'tcx>,
        tys: &[Ty<'tcx>],
    ) -> Type<'c> {
        let types = tys.iter().map(|ty| self.map_type(context, tcx, ty)).collect::<Vec<_>>();
        TupleType::new(context, &types).into()
    }

    fn create_raw_ptr_type<'tcx, 'c>(
        &self,
        context: &'c Context,
        tcx: &TyCtxt<'tcx>,
        ty: &Ty<'tcx>,
    ) -> Type<'c> {
        let ty = self.map_type(context, tcx, ty);
        pointer_type(ty)
    }
}


pub fn triton_tensor_handler<'tcx, 'c>(
    type_mapper: &TypeMapper,
    context: &'c Context,
    tcx: &TyCtxt<'tcx>,
    args: &[GenericArg<'tcx>],
) -> Type<'c> {
    debug_assert_eq!(args.len(), 1, "Tensor should have 1 argument");
    let arg_ty = args[0].expect_ty();
    let arg_type = type_mapper.map_type(context, tcx, &arg_ty);
    tensor_type(&[i64::MIN], arg_type).into()
}

pub fn triton_pointer_handler<'tcx, 'c>(
    type_mapper: &TypeMapper,
    context: &'c Context,
    tcx: &TyCtxt<'tcx>,
    args: &[GenericArg<'tcx>],
) -> Type<'c> {
    debug_assert_eq!(args.len(), 1, "Pointer should have 1 argument");
    let arg_ty = args[0].expect_ty();
    let arg_type = type_mapper.map_type(context, tcx, &arg_ty);
    pointer_type(arg_type)
}

pub fn triton_bool_handler<'tcx, 'c>(
    type_mapper: &TypeMapper,
    context: &'c Context,
    _tcx: &TyCtxt<'tcx>,
    _args: &[GenericArg<'tcx>],
) -> Type<'c> {
    type_mapper.create_bool_type(context)
}

pub fn triton_program_axis_handler<'tcx, 'c>(
    _type_mapper: &TypeMapper,
    context: &'c Context,
    _tcx: &TyCtxt<'tcx>,
    _args: &[GenericArg<'tcx>],
) -> Type<'c> {
    IntegerType::new(context, 32).into()
}
