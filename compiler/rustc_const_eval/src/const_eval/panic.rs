use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, layout::LayoutOf, subst::Subst};
use rustc_span::sym;
use std::cell::Cell;
use std::fmt::{self, Debug, Formatter};

use crate::interpret::{FnVal, InterpCx, InterpErrorInfo, InterpResult, OpTy};

use super::CompileTimeInterpreter;

struct Arg<'mir, 'tcx, 'err> {
    cx: &'err InterpCx<'mir, 'tcx, CompileTimeInterpreter<'mir, 'tcx>>,
    arg: OpTy<'tcx>,
    fmt_trait: DefId,
    err: &'err Cell<Option<InterpErrorInfo<'tcx>>>,
}

impl Debug for Arg<'_, '_, '_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.cx.fmt_arg(self.arg, self.fmt_trait, f) {
            Ok(_) => Ok(()),
            Err(e) => {
                self.err.set(Some(e));
                Err(fmt::Error)
            }
        }
    }
}

impl<'mir, 'tcx> InterpCx<'mir, 'tcx, CompileTimeInterpreter<'mir, 'tcx>> {
    fn fmt_arg(
        &self,
        arg: OpTy<'tcx>,
        fmt_trait: DefId,
        f: &mut Formatter<'_>,
    ) -> InterpResult<'tcx> {
        let fmt_trait_sym = self.tcx.item_name(fmt_trait);
        let fmt_trait_name = fmt_trait_sym.as_str();

        macro_rules! dispatch_fmt {
            ($e: expr, $($t: ident)|*) => {
                let _ = match &*fmt_trait_name {
                    $(stringify!($t) => fmt::$t::fmt($e, f),)*
                    _ => Debug::fmt($e, f),
                };
            }
        }

        match arg.layout.ty.kind() {
            ty::Bool => {
                let v = self.read_scalar(&arg)?.to_bool()?;
                dispatch_fmt!(&v, Display);
            }
            ty::Char => {
                let v = self.read_scalar(&arg)?.to_char()?;
                dispatch_fmt!(&v, Display);
            }
            ty::Int(int_ty) => {
                let v = self.read_scalar(&arg)?.check_init()?;
                let v = match int_ty {
                    ty::IntTy::I8 => v.to_i8()?.into(),
                    ty::IntTy::I16 => v.to_i16()?.into(),
                    ty::IntTy::I32 => v.to_i32()?.into(),
                    ty::IntTy::I64 => v.to_i64()?.into(),
                    ty::IntTy::I128 => v.to_i128()?,
                    ty::IntTy::Isize => v.to_machine_isize(self)?.into(),
                };
                dispatch_fmt!(
                    &v,
                    Display | Binary | Octal | LowerHex | UpperHex | LowerExp | UpperExp
                );
            }
            ty::Uint(int_ty) => {
                let v = self.read_scalar(&arg)?.check_init()?;
                let v = match int_ty {
                    ty::UintTy::U8 => v.to_u8()?.into(),
                    ty::UintTy::U16 => v.to_u16()?.into(),
                    ty::UintTy::U32 => v.to_u32()?.into(),
                    ty::UintTy::U64 => v.to_u64()?.into(),
                    ty::UintTy::U128 => v.to_u128()?,
                    ty::UintTy::Usize => v.to_machine_usize(self)?.into(),
                };
                dispatch_fmt!(
                    &v,
                    Display | Binary | Octal | LowerHex | UpperHex | LowerExp | UpperExp
                );
            }
            ty::Float(ty::FloatTy::F32) => {
                let v = f32::from_bits(self.read_scalar(&arg)?.to_u32()?);
                dispatch_fmt!(&v, Display);
            }
            ty::Float(ty::FloatTy::F64) => {
                let v = f64::from_bits(self.read_scalar(&arg)?.to_u64()?);
                dispatch_fmt!(&v, Display);
            }
            ty::Str => {
                let Ok(place) = arg.try_as_mplace() else {
                    bug!("str is not in MemPlace");
                };
                let v = self.read_str(&place)?;
                dispatch_fmt!(v, Display);
            }
            ty::Array(..) | ty::Slice(..) => {
                let Ok(place) = arg.try_as_mplace() else {
                    bug!("array/slice is not in MemPlace");
                };
                let err = Cell::new(None);
                let mut debug_list = f.debug_list();
                for field in self.mplace_array_fields(&place)? {
                    debug_list.entry(&Arg { cx: self, arg: field?.into(), fmt_trait, err: &err });
                }
                let _ = debug_list.finish();
                if let Some(e) = err.into_inner() {
                    return Err(e);
                }
            }
            ty::RawPtr(..) | ty::FnPtr(..) => {
                // This isn't precisely how Pointer is implemented, but it's best we can do.
                let ptr = self.read_pointer(&arg)?;
                let _ = write!(f, "{:?}", ptr);
            }
            ty::Tuple(substs) => {
                let err = Cell::new(None);
                let mut debug_tuple = f.debug_tuple("");
                for i in 0..substs.len() {
                    debug_tuple.field(&Arg {
                        cx: self,
                        arg: self.operand_field(&arg, i)?,
                        fmt_trait,
                        err: &err,
                    });
                }
                let _ = debug_tuple.finish();
                if let Some(e) = err.into_inner() {
                    return Err(e);
                }
            }

            // FIXME(nbdd0121): extend to allow fmt trait as super trait
            ty::Dynamic(list, _) if list.principal_def_id() == Some(fmt_trait) => {
                let Ok(place) = arg.try_as_mplace() else {
                    bug!("dyn is not in MemPlace");
                };
                let place = self.unpack_dyn_trait(&place)?.1;
                return self.fmt_arg(place.into(), fmt_trait, f);
            }

            ty::Ref(..) if fmt_trait_name == "Pointer" => {
                let ptr = self.read_pointer(&arg)?;
                let _ = write!(f, "{:?}", ptr);
            }
            ty::Ref(..) => {
                // FIXME(nbdd0121): User can implement trait on &UserType, so this isn't always correct.
                let place = self.deref_operand(&arg)?;
                return self.fmt_arg(place.into(), fmt_trait, f);
            }

            ty::Adt(adt, _) if self.tcx.is_diagnostic_item(sym::Arguments, adt.did) => {
                return self.fmt_arguments(arg, f);
            }

            // FIXME(nbdd0121): ty::Adt(..) => (),
            _ => {
                let _ = write!(f, "<failed to format {}>", arg.layout.ty);
            }
        }
        Ok(())
    }

    fn fmt_arguments(&self, arguments: OpTy<'tcx>, f: &mut Formatter<'_>) -> InterpResult<'tcx> {
        // Check we are dealing with the simple form
        let fmt_variant_idx = self.read_discriminant(&self.operand_field(&arguments, 1)?)?.1;
        if fmt_variant_idx.as_usize() != 0 {
            // FIXME(nbdd0121): implement complex format
            let _ = write!(f, "<cannot evaluate complex format>");
            return Ok(());
        }

        // `pieces: &[&str]`
        let pieces_place = self.deref_operand(&self.operand_field(&arguments, 0)?)?;
        let mut pieces = Vec::new();
        for piece in self.mplace_array_fields(&pieces_place)? {
            let piece: OpTy<'tcx> = piece?.into();
            pieces.push(self.read_str(&self.deref_operand(&piece)?)?);
        }

        // `args: &[ArgumentV1]`
        let args_place = self.deref_operand(&self.operand_field(&arguments, 2)?)?;
        let mut args = Vec::new();
        let err = Cell::new(None);
        for arg in self.mplace_array_fields(&args_place)? {
            let arg: OpTy<'tcx> = arg?.into();

            let fmt_fn = self.memory.get_fn(self.read_pointer(&self.operand_field(&arg, 1)?)?)?;
            let fmt_fn = match fmt_fn {
                FnVal::Instance(instance) => instance,
                FnVal::Other(o) => match o {},
            };

            // The formatter must an instance of fmt method of a fmt trait.
            let Some(fmt_impl) = self.tcx.impl_of_method(fmt_fn.def_id()) else {
                throw_unsup_format!("fmt function is not from trait impl")
            };
            let Some(fmt_trait) = self.tcx.impl_trait_ref(fmt_impl) else {
                throw_unsup_format!("fmt function is not from trait impl")
            };

            // Retrieve the trait ref with concrete self ty.
            let fmt_trait = fmt_trait.subst(*self.tcx, &fmt_fn.substs);

            // Change the opaque type into the actual type.
            let mut value_place = self.deref_operand(&self.operand_field(&arg, 0)?)?;
            value_place.layout = self.layout_of(fmt_trait.self_ty())?;

            args.push(Arg {
                cx: self,
                arg: value_place.into(),
                fmt_trait: fmt_trait.def_id,
                err: &err,
            });
        }

        // SAFETY: This transmutes `&[&str]` to `&[&'static str]` so it can be used in
        // `core::fmt::Arguments`. The slice will not be used after `write_fmt`.
        let static_pieces = unsafe { core::mem::transmute(&pieces[..]) };
        let arg_v1s = args.iter().map(|x| fmt::ArgumentV1::new(x, Debug::fmt)).collect::<Vec<_>>();
        let fmt_args = fmt::Arguments::new_v1(static_pieces, &arg_v1s);
        let _ = f.write_fmt(fmt_args);
        if let Some(v) = err.into_inner() {
            return Err(v);
        }
        Ok(())
    }

    pub(super) fn eval_const_panic_fmt(
        &mut self,
        arguments: OpTy<'tcx>,
    ) -> InterpResult<'tcx, String> {
        let mut msg = String::new();
        let mut formatter = Formatter::new(&mut msg);
        self.fmt_arguments(arguments, &mut formatter)?;
        Ok(msg)
    }
}
