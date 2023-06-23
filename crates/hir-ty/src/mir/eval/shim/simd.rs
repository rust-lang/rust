//! Shim implementation for simd intrinsics

use crate::TyKind;

use super::*;

macro_rules! from_bytes {
    ($ty:tt, $value:expr) => {
        ($ty::from_le_bytes(match ($value).try_into() {
            Ok(x) => x,
            Err(_) => return Err(MirEvalError::TypeError("mismatched size")),
        }))
    };
}

macro_rules! not_supported {
    ($x: expr) => {
        return Err(MirEvalError::NotSupported(format!($x)))
    };
}

impl Evaluator<'_> {
    fn detect_simd_ty(&self, ty: &Ty) -> Result<usize> {
        match ty.kind(Interner) {
            TyKind::Adt(_, subst) => {
                let Some(len) = subst.as_slice(Interner).get(1).and_then(|x| x.constant(Interner)) else {
                    return Err(MirEvalError::TypeError("simd type without len param"));
                };
                match try_const_usize(self.db, len) {
                    Some(x) => Ok(x as usize),
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
        _locals: &Locals<'_>,
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
                    .map(|(&x, &y)| match name {
                        "and" => x & y,
                        "or" => x | y,
                        "xor" => x ^ y,
                        _ => unreachable!(),
                    })
                    .collect::<Vec<_>>();
                destination.write_from_bytes(self, &result)
            }
            "eq" | "ne" => {
                let [left, right] = args else {
                    return Err(MirEvalError::TypeError("simd_eq args are not provided"));
                };
                let result = left.get(self)? == right.get(self)?;
                let result = result ^ (name == "ne");
                destination.write_from_bytes(self, &[u8::from(result)])
            }
            "bitmask" => {
                let [op] = args else {
                    return Err(MirEvalError::TypeError("simd_shuffle args are not provided"));
                };
                let op_len = self.detect_simd_ty(&op.ty)?;
                let op_count = op.interval.size / op_len;
                let mut result: u64 = 0;
                for (i, val) in op.get(self)?.chunks(op_count).enumerate() {
                    if !val.iter().all(|&x| x == 0) {
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
                    return Err(MirEvalError::TypeError("simd_shuffle index argument has non-array type"));
                };
                let index_len = match try_const_usize(self.db, index_len) {
                    Some(x) => x as usize,
                    None => {
                        return Err(MirEvalError::TypeError(
                            "simd type with unevaluatable len param",
                        ))
                    }
                };
                let left_len = self.detect_simd_ty(&left.ty)?;
                let left_count = left.interval.size / left_len;
                let vector =
                    left.get(self)?.chunks(left_count).chain(right.get(self)?.chunks(left_count));
                let mut result = vec![];
                for index in index.get(self)?.chunks(index.interval.size / index_len) {
                    let index = from_bytes!(u32, index) as usize;
                    let val = match vector.clone().nth(index) {
                        Some(x) => x,
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
