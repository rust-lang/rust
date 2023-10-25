use rustc_data_structures::stable_hasher::HashStable;
use rustc_serialize::{Decodable, Decoder, Encodable};
use std::cmp::Ordering;
use std::fmt;
use std::hash;

use crate::{
    DebruijnIndex, DebugWithInfcx, HashStableContext, InferCtxtLike, Interner, TyDecoder,
    TyEncoder, WithInfcx,
};

use self::RegionKind::*;

/// Representation of regions. Note that the NLL checker uses a distinct
/// representation of regions. For this reason, it internally replaces all the
/// regions with inference variables -- the index of the variable is then used
/// to index into internal NLL data structures. See `rustc_const_eval::borrow_check`
/// module for more information.
///
/// Note: operations are on the wrapper `Region` type, which is interned,
/// rather than this type.
///
/// ## The Region lattice within a given function
///
/// In general, the region lattice looks like
///
/// ```text
/// static ----------+-----...------+       (greatest)
/// |                |              |
/// early-bound and  |              |
/// free regions     |              |
/// |                |              |
/// |                |              |
/// empty(root)   placeholder(U1)   |
/// |            /                  |
/// |           /         placeholder(Un)
/// empty(U1) --         /
/// |                   /
/// ...                /
/// |                 /
/// empty(Un) --------                      (smallest)
/// ```
///
/// Early-bound/free regions are the named lifetimes in scope from the
/// function declaration. They have relationships to one another
/// determined based on the declared relationships from the
/// function.
///
/// Note that inference variables and bound regions are not included
/// in this diagram. In the case of inference variables, they should
/// be inferred to some other region from the diagram. In the case of
/// bound regions, they are excluded because they don't make sense to
/// include -- the diagram indicates the relationship between free
/// regions.
///
/// ## Inference variables
///
/// During region inference, we sometimes create inference variables,
/// represented as `ReVar`. These will be inferred by the code in
/// `infer::lexical_region_resolve` to some free region from the
/// lattice above (the minimal region that meets the
/// constraints).
///
/// During NLL checking, where regions are defined differently, we
/// also use `ReVar` -- in that case, the index is used to index into
/// the NLL region checker's data structures. The variable may in fact
/// represent either a free region or an inference variable, in that
/// case.
///
/// ## Bound Regions
///
/// These are regions that are stored behind a binder and must be substituted
/// with some concrete region before being used. There are two kind of
/// bound regions: early-bound, which are bound in an item's `Generics`,
/// and are substituted by an `GenericArgs`, and late-bound, which are part of
/// higher-ranked types (e.g., `for<'a> fn(&'a ())`), and are substituted by
/// the likes of `liberate_late_bound_regions`. The distinction exists
/// because higher-ranked lifetimes aren't supported in all places. See [1][2].
///
/// Unlike `Param`s, bound regions are not supposed to exist "in the wild"
/// outside their binder, e.g., in types passed to type inference, and
/// should first be substituted (by placeholder regions, free regions,
/// or region variables).
///
/// ## Placeholder and Free Regions
///
/// One often wants to work with bound regions without knowing their precise
/// identity. For example, when checking a function, the lifetime of a borrow
/// can end up being assigned to some region parameter. In these cases,
/// it must be ensured that bounds on the region can't be accidentally
/// assumed without being checked.
///
/// To do this, we replace the bound regions with placeholder markers,
/// which don't satisfy any relation not explicitly provided.
///
/// There are two kinds of placeholder regions in rustc: `ReFree` and
/// `RePlaceholder`. When checking an item's body, `ReFree` is supposed
/// to be used. These also support explicit bounds: both the internally-stored
/// *scope*, which the region is assumed to outlive, as well as other
/// relations stored in the `FreeRegionMap`. Note that these relations
/// aren't checked when you `make_subregion` (or `eq_types`), only by
/// `resolve_regions_and_report_errors`.
///
/// When working with higher-ranked types, some region relations aren't
/// yet known, so you can't just call `resolve_regions_and_report_errors`.
/// `RePlaceholder` is designed for this purpose. In these contexts,
/// there's also the risk that some inference variable laying around will
/// get unified with your placeholder region: if you want to check whether
/// `for<'a> Foo<'_>: 'a`, and you substitute your bound region `'a`
/// with a placeholder region `'%a`, the variable `'_` would just be
/// instantiated to the placeholder region `'%a`, which is wrong because
/// the inference variable is supposed to satisfy the relation
/// *for every value of the placeholder region*. To ensure that doesn't
/// happen, you can use `leak_check`. This is more clearly explained
/// by the [rustc dev guide].
///
/// [1]: https://smallcultfollowing.com/babysteps/blog/2013/10/29/intermingled-parameter-lists/
/// [2]: https://smallcultfollowing.com/babysteps/blog/2013/11/04/intermingled-parameter-lists/
/// [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/traits/hrtb.html
pub enum RegionKind<I: Interner> {
    /// Region bound in a type or fn declaration which will be
    /// substituted 'early' -- that is, at the same time when type
    /// parameters are substituted.
    ReEarlyBound(I::EarlyBoundRegion),

    /// Region bound in a function scope, which will be substituted when the
    /// function is called.
    ReLateBound(DebruijnIndex, I::BoundRegion),

    /// When checking a function body, the types of all arguments and so forth
    /// that refer to bound region parameters are modified to refer to free
    /// region parameters.
    ReFree(I::FreeRegion),

    /// Static data that has an "infinite" lifetime. Top in the region lattice.
    ReStatic,

    /// A region variable. Should not exist outside of type inference.
    ReVar(I::InferRegion),

    /// A placeholder region -- basically, the higher-ranked version of `ReFree`.
    /// Should not exist outside of type inference.
    RePlaceholder(I::PlaceholderRegion),

    /// Erased region, used by trait selection, in MIR and during codegen.
    ReErased,

    /// A region that resulted from some other error. Used exclusively for diagnostics.
    ReError(I::ErrorGuaranteed),
}

// This is manually implemented for `RegionKind` because `std::mem::discriminant`
// returns an opaque value that is `PartialEq` but not `PartialOrd`
#[inline]
const fn regionkind_discriminant<I: Interner>(value: &RegionKind<I>) -> usize {
    match value {
        ReEarlyBound(_) => 0,
        ReLateBound(_, _) => 1,
        ReFree(_) => 2,
        ReStatic => 3,
        ReVar(_) => 4,
        RePlaceholder(_) => 5,
        ReErased => 6,
        ReError(_) => 7,
    }
}

// This is manually implemented because a derive would require `I: Copy`
impl<I: Interner> Copy for RegionKind<I>
where
    I::EarlyBoundRegion: Copy,
    I::BoundRegion: Copy,
    I::FreeRegion: Copy,
    I::InferRegion: Copy,
    I::PlaceholderRegion: Copy,
    I::ErrorGuaranteed: Copy,
{
}

// This is manually implemented because a derive would require `I: Clone`
impl<I: Interner> Clone for RegionKind<I> {
    fn clone(&self) -> Self {
        match self {
            ReEarlyBound(r) => ReEarlyBound(r.clone()),
            ReLateBound(d, r) => ReLateBound(*d, r.clone()),
            ReFree(r) => ReFree(r.clone()),
            ReStatic => ReStatic,
            ReVar(r) => ReVar(r.clone()),
            RePlaceholder(r) => RePlaceholder(r.clone()),
            ReErased => ReErased,
            ReError(r) => ReError(r.clone()),
        }
    }
}

// This is manually implemented because a derive would require `I: PartialEq`
impl<I: Interner> PartialEq for RegionKind<I> {
    #[inline]
    fn eq(&self, other: &RegionKind<I>) -> bool {
        regionkind_discriminant(self) == regionkind_discriminant(other)
            && match (self, other) {
                (ReEarlyBound(a_r), ReEarlyBound(b_r)) => a_r == b_r,
                (ReLateBound(a_d, a_r), ReLateBound(b_d, b_r)) => a_d == b_d && a_r == b_r,
                (ReFree(a_r), ReFree(b_r)) => a_r == b_r,
                (ReStatic, ReStatic) => true,
                (ReVar(a_r), ReVar(b_r)) => a_r == b_r,
                (RePlaceholder(a_r), RePlaceholder(b_r)) => a_r == b_r,
                (ReErased, ReErased) => true,
                (ReError(_), ReError(_)) => true,
                _ => {
                    debug_assert!(
                        false,
                        "This branch must be unreachable, maybe the match is missing an arm? self = {self:?}, other = {other:?}"
                    );
                    true
                }
            }
    }
}

// This is manually implemented because a derive would require `I: Eq`
impl<I: Interner> Eq for RegionKind<I> {}

// This is manually implemented because a derive would require `I: PartialOrd`
impl<I: Interner> PartialOrd for RegionKind<I> {
    #[inline]
    fn partial_cmp(&self, other: &RegionKind<I>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// This is manually implemented because a derive would require `I: Ord`
impl<I: Interner> Ord for RegionKind<I> {
    #[inline]
    fn cmp(&self, other: &RegionKind<I>) -> Ordering {
        regionkind_discriminant(self).cmp(&regionkind_discriminant(other)).then_with(|| {
            match (self, other) {
                (ReEarlyBound(a_r), ReEarlyBound(b_r)) => a_r.cmp(b_r),
                (ReLateBound(a_d, a_r), ReLateBound(b_d, b_r)) => {
                    a_d.cmp(b_d).then_with(|| a_r.cmp(b_r))
                }
                (ReFree(a_r), ReFree(b_r)) => a_r.cmp(b_r),
                (ReStatic, ReStatic) => Ordering::Equal,
                (ReVar(a_r), ReVar(b_r)) => a_r.cmp(b_r),
                (RePlaceholder(a_r), RePlaceholder(b_r)) => a_r.cmp(b_r),
                (ReErased, ReErased) => Ordering::Equal,
                _ => {
                    debug_assert!(false, "This branch must be unreachable, maybe the match is missing an arm? self = self = {self:?}, other = {other:?}");
                    Ordering::Equal
                }
            }
        })
    }
}

// This is manually implemented because a derive would require `I: Hash`
impl<I: Interner> hash::Hash for RegionKind<I> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) -> () {
        regionkind_discriminant(self).hash(state);
        match self {
            ReEarlyBound(r) => r.hash(state),
            ReLateBound(d, r) => {
                d.hash(state);
                r.hash(state)
            }
            ReFree(r) => r.hash(state),
            ReStatic => (),
            ReVar(r) => r.hash(state),
            RePlaceholder(r) => r.hash(state),
            ReErased => (),
            ReError(_) => (),
        }
    }
}

impl<I: Interner> DebugWithInfcx<I> for RegionKind<I> {
    fn fmt<Infcx: InferCtxtLike<Interner = I>>(
        this: WithInfcx<'_, Infcx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        match this.data {
            ReEarlyBound(data) => write!(f, "ReEarlyBound({data:?})"),

            ReLateBound(binder_id, bound_region) => {
                write!(f, "ReLateBound({binder_id:?}, {bound_region:?})")
            }

            ReFree(fr) => write!(f, "{fr:?}"),

            ReStatic => f.write_str("ReStatic"),

            ReVar(vid) => write!(f, "{:?}", &this.wrap(vid)),

            RePlaceholder(placeholder) => write!(f, "RePlaceholder({placeholder:?})"),

            ReErased => f.write_str("ReErased"),

            ReError(_) => f.write_str("ReError"),
        }
    }
}
impl<I: Interner> fmt::Debug for RegionKind<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        WithInfcx::with_no_infcx(self).fmt(f)
    }
}

// This is manually implemented because a derive would require `I: Encodable`
impl<I: Interner, E: TyEncoder> Encodable<E> for RegionKind<I>
where
    I::EarlyBoundRegion: Encodable<E>,
    I::BoundRegion: Encodable<E>,
    I::FreeRegion: Encodable<E>,
    I::InferRegion: Encodable<E>,
    I::PlaceholderRegion: Encodable<E>,
{
    fn encode(&self, e: &mut E) {
        let disc = regionkind_discriminant(self);
        match self {
            ReEarlyBound(a) => e.emit_enum_variant(disc, |e| {
                a.encode(e);
            }),
            ReLateBound(a, b) => e.emit_enum_variant(disc, |e| {
                a.encode(e);
                b.encode(e);
            }),
            ReFree(a) => e.emit_enum_variant(disc, |e| {
                a.encode(e);
            }),
            ReStatic => e.emit_enum_variant(disc, |_| {}),
            ReVar(a) => e.emit_enum_variant(disc, |e| {
                a.encode(e);
            }),
            RePlaceholder(a) => e.emit_enum_variant(disc, |e| {
                a.encode(e);
            }),
            ReErased => e.emit_enum_variant(disc, |_| {}),
            ReError(_) => e.emit_enum_variant(disc, |_| {}),
        }
    }
}

// This is manually implemented because a derive would require `I: Decodable`
impl<I: Interner, D: TyDecoder<I = I>> Decodable<D> for RegionKind<I>
where
    I::EarlyBoundRegion: Decodable<D>,
    I::BoundRegion: Decodable<D>,
    I::FreeRegion: Decodable<D>,
    I::InferRegion: Decodable<D>,
    I::PlaceholderRegion: Decodable<D>,
    I::ErrorGuaranteed: Decodable<D>,
{
    fn decode(d: &mut D) -> Self {
        match Decoder::read_usize(d) {
            0 => ReEarlyBound(Decodable::decode(d)),
            1 => ReLateBound(Decodable::decode(d), Decodable::decode(d)),
            2 => ReFree(Decodable::decode(d)),
            3 => ReStatic,
            4 => ReVar(Decodable::decode(d)),
            5 => RePlaceholder(Decodable::decode(d)),
            6 => ReErased,
            7 => ReError(Decodable::decode(d)),
            _ => panic!(
                "{}",
                format!(
                    "invalid enum variant tag while decoding `{}`, expected 0..{}",
                    "RegionKind", 8,
                )
            ),
        }
    }
}

// This is not a derived impl because a derive would require `I: HashStable`
impl<CTX: HashStableContext, I: Interner> HashStable<CTX> for RegionKind<I>
where
    I::EarlyBoundRegion: HashStable<CTX>,
    I::BoundRegion: HashStable<CTX>,
    I::FreeRegion: HashStable<CTX>,
    I::InferRegion: HashStable<CTX>,
    I::PlaceholderRegion: HashStable<CTX>,
{
    #[inline]
    fn hash_stable(
        &self,
        hcx: &mut CTX,
        hasher: &mut rustc_data_structures::stable_hasher::StableHasher,
    ) {
        std::mem::discriminant(self).hash_stable(hcx, hasher);
        match self {
            ReErased | ReStatic | ReError(_) => {
                // No variant fields to hash for these ...
            }
            ReLateBound(d, r) => {
                d.hash_stable(hcx, hasher);
                r.hash_stable(hcx, hasher);
            }
            ReEarlyBound(r) => {
                r.hash_stable(hcx, hasher);
            }
            ReFree(r) => {
                r.hash_stable(hcx, hasher);
            }
            RePlaceholder(r) => {
                r.hash_stable(hcx, hasher);
            }
            ReVar(_) => {
                panic!("region variables should not be hashed: {self:?}")
            }
        }
    }
}
