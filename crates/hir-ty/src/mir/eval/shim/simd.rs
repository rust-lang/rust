//! Shim implementation for simd intrinsics

use std::cmp::Ordering;

use crate::TyKind;

use super::*;

macro_rules! from_bytes {
    ($ty:tt, $value:expr) => {
        ($ty::from_le_bytes(match ($value).try_into() {
            Ok(it) => it,
            Err(_) => return Err(MirEvalError::TypeError("mismatched size")),
        }))
    };
}

macro_rules! not_supported {
    ($it: expr) => {
        return Err(MirEvalError::NotSupported(format!($it)))
    };
}

impl Evaluator<'_> {
    fn detect_simd_ty(&self, ty: &Ty) -> Result<usize> {
        match ty.kind(Interner) {
            TyKind::Adt(id, subst) => {
                let len = match subst.as_slice(Interner).get(1).and_then(|it| it.constant(Interner))
                {
                    Some(len) => len,
                    _ => {
                        if let AdtId::StructId(id) = id.0 {
                            return Ok(self.db.struct_data(id).variant_data.fields().len());
                        }
                        return Err(MirEvalError::TypeError("simd type with no len param"));
                    }
                };
                match try_const_usize(self.db, len) {
                    Some(it) => Ok(it as usize),
                    None => Err(MirEvalError::TypeError("simd type with unevaluatable len param")),
                }
            }
            _ => Err(MirEvalError::TypeError("simd type which is not a struct")),
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
                    return Err(MirEvalError::TypeError("simd bit op args are not provided"));
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
                    return Err(MirEvalError::TypeError("simd args are not provided"));
                };
                let len = self.detect_simd_ty(&left.ty)?;
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
                    let result = match result {
                        Ordering::Less => ["lt", "le", "ne"].contains(&name),
                        Ordering::Equal => ["ge", "le", "eq"].contains(&name),
                        Ordering::Greater => ["ge", "gt", "ne"].contains(&name),
                    };
                    let result = if result { 255 } else { 0 };
                    destination_bytes.extend(std::iter::repeat(result).take(dest_size));
                }

                destination.write_from_bytes(self, &destination_bytes)
            }
            "bitmask" => {
                let [op] = args else {
                    return Err(MirEvalError::TypeError("simd_shuffle args are not provided"));
                };
                let op_len = self.detect_simd_ty(&op.ty)?;
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
                    return Err(MirEvalError::TypeError("simd_shuffle args are not provided"));
                };
                let TyKind::Array(_, index_len) = index.ty.kind(Interner) else {
                    return Err(MirEvalError::TypeError(
                        "simd_shuffle index argument has non-array type",
                    ));
                };
                let index_len = match try_const_usize(self.db, index_len) {
                    Some(it) => it as usize,
                    None => {
                        return Err(MirEvalError::TypeError(
                            "simd type with unevaluatable len param",
                        ))
                    }
                };
                let left_len = self.detect_simd_ty(&left.ty)?;
                let left_size = left.interval.size / left_len;
                let vector =
                    left.get(self)?.chunks(left_size).chain(right.get(self)?.chunks(left_size));
                let mut result = vec![];
                for index in index.get(self)?.chunks(index.interval.size / index_len) {
                    let index = from_bytes!(u32, index) as usize;
                    let val = match vector.clone().nth(index) {
                        Some(it) => it,
                        None => {
                            return Err(MirEvalError::TypeError(
                                "out of bound access in simd shuffle",
                            ))
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
