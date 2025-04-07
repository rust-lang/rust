#[cfg(feature = "nightly")]
use crate::{BackendRepr, FieldsShape, Primitive, Size, TyAbiInterface, TyAndLayout, Variants};

mod reg;

pub use reg::{Reg, RegKind};

/// Return value from the `homogeneous_aggregate` test function.
#[derive(Copy, Clone, Debug)]
pub enum HomogeneousAggregate {
    /// Yes, all the "leaf fields" of this struct are passed in the
    /// same way (specified in the `Reg` value).
    Homogeneous(Reg),

    /// There are no leaf fields at all.
    NoData,
}

/// Error from the `homogeneous_aggregate` test function, indicating
/// there are distinct leaf fields passed in different ways,
/// or this is uninhabited.
#[derive(Copy, Clone, Debug)]
pub struct Heterogeneous;

impl HomogeneousAggregate {
    /// If this is a homogeneous aggregate, returns the homogeneous
    /// unit, else `None`.
    pub fn unit(self) -> Option<Reg> {
        match self {
            HomogeneousAggregate::Homogeneous(reg) => Some(reg),
            HomogeneousAggregate::NoData => None,
        }
    }

    /// Try to combine two `HomogeneousAggregate`s, e.g. from two fields in
    /// the same `struct`. Only succeeds if only one of them has any data,
    /// or both units are identical.
    fn merge(self, other: HomogeneousAggregate) -> Result<HomogeneousAggregate, Heterogeneous> {
        match (self, other) {
            (x, HomogeneousAggregate::NoData) | (HomogeneousAggregate::NoData, x) => Ok(x),

            (HomogeneousAggregate::Homogeneous(a), HomogeneousAggregate::Homogeneous(b)) => {
                if a != b {
                    return Err(Heterogeneous);
                }
                Ok(self)
            }
        }
    }
}

#[cfg(feature = "nightly")]
impl<'a, Ty> TyAndLayout<'a, Ty> {
    /// Returns `Homogeneous` if this layout is an aggregate containing fields of
    /// only a single type (e.g., `(u32, u32)`). Such aggregates are often
    /// special-cased in ABIs.
    ///
    /// Note: We generally ignore 1-ZST fields when computing this value (see #56877).
    ///
    /// This is public so that it can be used in unit tests, but
    /// should generally only be relevant to the ABI details of
    /// specific targets.
    pub fn homogeneous_aggregate<C>(&self, cx: &C) -> Result<HomogeneousAggregate, Heterogeneous>
    where
        Ty: TyAbiInterface<'a, C> + Copy,
    {
        match self.backend_repr {
            // The primitive for this algorithm.
            BackendRepr::Scalar(scalar) => {
                let kind = match scalar.primitive() {
                    Primitive::Int(..) | Primitive::Pointer(_) => RegKind::Integer,
                    Primitive::Float(_) => RegKind::Float,
                };
                Ok(HomogeneousAggregate::Homogeneous(Reg { kind, size: self.size }))
            }

            BackendRepr::SimdVector { .. } => {
                assert!(!self.is_zst());
                Ok(HomogeneousAggregate::Homogeneous(Reg {
                    kind: RegKind::Vector,
                    size: self.size,
                }))
            }

            BackendRepr::ScalarPair(..) | BackendRepr::Memory { sized: true } => {
                // Helper for computing `homogeneous_aggregate`, allowing a custom
                // starting offset (used below for handling variants).
                let from_fields_at =
                    |layout: Self,
                     start: Size|
                     -> Result<(HomogeneousAggregate, Size), Heterogeneous> {
                        let is_union = match layout.fields {
                            FieldsShape::Primitive => {
                                unreachable!("aggregates can't have `FieldsShape::Primitive`")
                            }
                            FieldsShape::Array { count, .. } => {
                                assert_eq!(start, Size::ZERO);

                                let result = if count > 0 {
                                    layout.field(cx, 0).homogeneous_aggregate(cx)?
                                } else {
                                    HomogeneousAggregate::NoData
                                };
                                return Ok((result, layout.size));
                            }
                            FieldsShape::Union(_) => true,
                            FieldsShape::Arbitrary { .. } => false,
                        };

                        let mut result = HomogeneousAggregate::NoData;
                        let mut total = start;

                        for i in 0..layout.fields.count() {
                            let field = layout.field(cx, i);
                            if field.is_1zst() {
                                // No data here and no impact on layout, can be ignored.
                                // (We might be able to also ignore all aligned ZST but that's less clear.)
                                continue;
                            }

                            if !is_union && total != layout.fields.offset(i) {
                                // This field isn't just after the previous one we considered, abort.
                                return Err(Heterogeneous);
                            }

                            result = result.merge(field.homogeneous_aggregate(cx)?)?;

                            // Keep track of the offset (without padding).
                            let size = field.size;
                            if is_union {
                                total = total.max(size);
                            } else {
                                total += size;
                            }
                        }

                        Ok((result, total))
                    };

                let (mut result, mut total) = from_fields_at(*self, Size::ZERO)?;

                match &self.variants {
                    Variants::Single { .. } | Variants::Empty => {}
                    Variants::Multiple { variants, .. } => {
                        // Treat enum variants like union members.
                        // HACK(eddyb) pretend the `enum` field (discriminant)
                        // is at the start of every variant (otherwise the gap
                        // at the start of all variants would disqualify them).
                        //
                        // NB: for all tagged `enum`s (which include all non-C-like
                        // `enum`s with defined FFI representation), this will
                        // match the homogeneous computation on the equivalent
                        // `struct { tag; union { variant1; ... } }` and/or
                        // `union { struct { tag; variant1; } ... }`
                        // (the offsets of variant fields should be identical
                        // between the two for either to be a homogeneous aggregate).
                        let variant_start = total;
                        for variant_idx in variants.indices() {
                            let (variant_result, variant_total) =
                                from_fields_at(self.for_variant(cx, variant_idx), variant_start)?;

                            result = result.merge(variant_result)?;
                            total = total.max(variant_total);
                        }
                    }
                }

                // There needs to be no padding.
                if total != self.size {
                    Err(Heterogeneous)
                } else {
                    match result {
                        HomogeneousAggregate::Homogeneous(_) => {
                            assert_ne!(total, Size::ZERO);
                        }
                        HomogeneousAggregate::NoData => {
                            assert_eq!(total, Size::ZERO);
                        }
                    }
                    Ok(result)
                }
            }
            BackendRepr::Memory { sized: false } => Err(Heterogeneous),
        }
    }
}
