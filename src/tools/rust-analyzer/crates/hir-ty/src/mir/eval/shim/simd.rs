//! Shim implementation for simd intrinsics

use std::cmp::Ordering;

use crate::TyKind;
use crate::consteval::try_const_usize;

use super::*;

macro_rules! from_bytes {
    ($ty:tt, $value:expr) => {
        ($ty::from_le_bytes(match ($value).try_into() {
            Ok(it) => it,
            Err(_) => return Err(MirEvalError::InternalError("mismatched size".into())),
        }))
    };
}

macro_rules! not_supported {
    ($it: expr) => {
        return Err(MirEvalError::NotSupported(format!($it)))
    };
}

impl Evaluator<'_> {
    fn detect_simd_ty(&self, ty: &Ty) -> Result<(usize, Ty)> {
        match ty.kind(Interner) {
            TyKind::Adt(id, subst) => {
                let len = match subst.as_slice(Interner).get(1).and_then(|it| it.constant(Interner))
                {
                    Some(len) => len,
                    _ => {
                        if let AdtId::StructId(id) = id.0 {
                            let struct_data = id.fields(self.db);
                            let fields = struct_data.fields();
                            let Some((first_field, _)) = fields.iter().next() else {
                                not_supported!("simd type with no field");
                            };
                            let field_ty = self.db.field_types(id.into())[first_field]
                                .clone()
                                .substitute(Interner, subst);
                            return Ok((fields.len(), field_ty));
                        }
                        return Err(MirEvalError::InternalError(
                            "simd type with no len param".into(),
                        ));
                    }
                };
                match try_const_usize(self.db, len) {
                    Some(len) => {
                        let Some(ty) =
                            subst.as_slice(Interner).first().and_then(|it| it.ty(Interner))
                        else {
                            return Err(MirEvalError::InternalError(
                                "simd type with no ty param".into(),
                            ));
                        };
                        Ok((len as usize, ty.clone()))
                    }
                    None => Err(MirEvalError::InternalError(
                        "simd type with unevaluatable len param".into(),
                    )),
                }
            }
            _ => Err(MirEvalError::InternalError("simd type which is not a struct".into())),
        }
    }

    pub(super) fn exec_simd_intrinsic(
        &mut self,
        name: &str,
        args: &[IntervalAndTy],
        _generic_args: &Substitution,
        destination: Interval,
        _locals: &Locals,
        _span: MirSpan,
    ) -> Result<()> {
        match name {
            "and" | "or" | "xor" => {
                let [left, right] = args else {
                    return Err(MirEvalError::InternalError(
                        "simd bit op args are not provided".into(),
                    ));
                };
                let result = left
                    .get(self)?
                    .iter()
                    .zip(right.get(self)?)
                    .map(|(&it, &y)| match name {
                        "and" => it & y,
                        "or" => it | y,
                        "xor" => it ^ y,
                        _ => unreachable!(),
                    })
                    .collect::<Vec<_>>();
                destination.write_from_bytes(self, &result)
            }
            "eq" | "ne" | "lt" | "le" | "gt" | "ge" => {
                let [left, right] = args else {
                    return Err(MirEvalError::InternalError("simd args are not provided".into()));
                };
                let (len, ty) = self.detect_simd_ty(&left.ty)?;
                let is_signed = matches!(ty.as_builtin(), Some(BuiltinType::Int(_)));
                let size = left.interval.size / len;
                let dest_size = destination.size / len;
                let mut destination_bytes = vec![];
                let vector = left.get(self)?.chunks(size).zip(right.get(self)?.chunks(size));
                for (l, r) in vector {
                    let mut result = Ordering::Equal;
                    for (l, r) in l.iter().zip(r).rev() {
                        let it = l.cmp(r);
                        if it != Ordering::Equal {
                            result = it;
                            break;
                        }
                    }
                    if is_signed
                        && let Some((&l, &r)) = l.iter().zip(r).next_back()
                        && l != r
                    {
                        result = (l as i8).cmp(&(r as i8));
                    }
                    let result = match result {
                        Ordering::Less => ["lt", "le", "ne"].contains(&name),
                        Ordering::Equal => ["ge", "le", "eq"].contains(&name),
                        Ordering::Greater => ["ge", "gt", "ne"].contains(&name),
                    };
                    let result = if result { 255 } else { 0 };
                    destination_bytes.extend(std::iter::repeat_n(result, dest_size));
                }

                destination.write_from_bytes(self, &destination_bytes)
            }
            "bitmask" => {
                let [op] = args else {
                    return Err(MirEvalError::InternalError(
                        "simd_bitmask args are not provided".into(),
                    ));
                };
                let (op_len, _) = self.detect_simd_ty(&op.ty)?;
                let op_count = op.interval.size / op_len;
                let mut result: u64 = 0;
                for (i, val) in op.get(self)?.chunks(op_count).enumerate() {
                    if !val.iter().all(|&it| it == 0) {
                        result |= 1 << i;
                    }
                }
                destination.write_from_bytes(self, &result.to_le_bytes()[0..destination.size])
            }
            "shuffle" => {
                let [left, right, index] = args else {
                    return Err(MirEvalError::InternalError(
                        "simd_shuffle args are not provided".into(),
                    ));
                };
                let TyKind::Array(_, index_len) = index.ty.kind(Interner) else {
                    return Err(MirEvalError::InternalError(
                        "simd_shuffle index argument has non-array type".into(),
                    ));
                };
                let index_len = match try_const_usize(self.db, index_len) {
                    Some(it) => it as usize,
                    None => {
                        return Err(MirEvalError::InternalError(
                            "simd type with unevaluatable len param".into(),
                        ));
                    }
                };
                let (left_len, _) = self.detect_simd_ty(&left.ty)?;
                let left_size = left.interval.size / left_len;
                let vector =
                    left.get(self)?.chunks(left_size).chain(right.get(self)?.chunks(left_size));
                let mut result = vec![];
                for index in index.get(self)?.chunks(index.interval.size / index_len) {
                    let index = from_bytes!(u32, index) as usize;
                    let val = match vector.clone().nth(index) {
                        Some(it) => it,
                        None => {
                            return Err(MirEvalError::InternalError(
                                "out of bound access in simd shuffle".into(),
                            ));
                        }
                    };
                    result.extend(val);
                }
                destination.write_from_bytes(self, &result)
            }
            _ => not_supported!("unknown simd intrinsic {name}"),
        }
    }
}
