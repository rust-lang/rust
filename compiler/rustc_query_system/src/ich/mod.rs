//! ICH - Incremental Compilation Hash

pub use self::hcx::StableHashingContext;
use rustc_data_structures::stable_hasher::HashStable;
use rustc_span::symbol::{sym, Symbol};

mod hcx;
mod impls_hir;
mod impls_syntax;

pub const IGNORED_ATTRIBUTES: &[Symbol] = &[
    sym::cfg,
    sym::rustc_if_this_changed,
    sym::rustc_then_this_would_need,
    sym::rustc_dirty,
    sym::rustc_clean,
    sym::rustc_partition_reused,
    sym::rustc_partition_codegened,
    sym::rustc_expected_cgu_reuse,
];

#[allow(rustc::usage_of_ty_tykind)]
impl<'__ctx, I: rustc_type_ir::Interner> HashStable<StableHashingContext<'__ctx>>
    for rustc_type_ir::TyKind<I>
where
    I::AdtDef: HashStable<StableHashingContext<'__ctx>>,
    I::DefId: HashStable<StableHashingContext<'__ctx>>,
    I::SubstsRef: HashStable<StableHashingContext<'__ctx>>,
    I::Ty: HashStable<StableHashingContext<'__ctx>>,
    I::Const: HashStable<StableHashingContext<'__ctx>>,
    I::TypeAndMut: HashStable<StableHashingContext<'__ctx>>,
    I::PolyFnSig: HashStable<StableHashingContext<'__ctx>>,
    I::ListBinderExistentialPredicate: HashStable<StableHashingContext<'__ctx>>,
    I::Region: HashStable<StableHashingContext<'__ctx>>,
    I::Movability: HashStable<StableHashingContext<'__ctx>>,
    I::Mutability: HashStable<StableHashingContext<'__ctx>>,
    I::BinderListTy: HashStable<StableHashingContext<'__ctx>>,
    I::ListTy: HashStable<StableHashingContext<'__ctx>>,
    I::ProjectionTy: HashStable<StableHashingContext<'__ctx>>,
    I::BoundTy: HashStable<StableHashingContext<'__ctx>>,
    I::ParamTy: HashStable<StableHashingContext<'__ctx>>,
    I::PlaceholderType: HashStable<StableHashingContext<'__ctx>>,
    I::InferTy: HashStable<StableHashingContext<'__ctx>>,
    I::DelaySpanBugEmitted: HashStable<StableHashingContext<'__ctx>>,
{
    #[inline]
    fn hash_stable(
        &self,
        __hcx: &mut crate::ich::StableHashingContext<'__ctx>,
        __hasher: &mut rustc_data_structures::stable_hasher::StableHasher,
    ) {
        std::mem::discriminant(self).hash_stable(__hcx, __hasher);
        use rustc_type_ir::TyKind::*;
        match self {
            Bool => {}
            Char => {}
            Int(i) => {
                i.hash_stable(__hcx, __hasher);
            }
            Uint(u) => {
                u.hash_stable(__hcx, __hasher);
            }
            Float(f) => {
                f.hash_stable(__hcx, __hasher);
            }
            Adt(adt, substs) => {
                adt.hash_stable(__hcx, __hasher);
                substs.hash_stable(__hcx, __hasher);
            }
            Foreign(def_id) => {
                def_id.hash_stable(__hcx, __hasher);
            }
            Str => {}
            Array(t, c) => {
                t.hash_stable(__hcx, __hasher);
                c.hash_stable(__hcx, __hasher);
            }
            Slice(t) => {
                t.hash_stable(__hcx, __hasher);
            }
            RawPtr(tam) => {
                tam.hash_stable(__hcx, __hasher);
            }
            Ref(r, t, m) => {
                r.hash_stable(__hcx, __hasher);
                t.hash_stable(__hcx, __hasher);
                m.hash_stable(__hcx, __hasher);
            }
            FnDef(def_id, substs) => {
                def_id.hash_stable(__hcx, __hasher);
                substs.hash_stable(__hcx, __hasher);
            }
            FnPtr(polyfnsig) => {
                polyfnsig.hash_stable(__hcx, __hasher);
            }
            Dynamic(l, r) => {
                l.hash_stable(__hcx, __hasher);
                r.hash_stable(__hcx, __hasher);
            }
            Closure(def_id, substs) => {
                def_id.hash_stable(__hcx, __hasher);
                substs.hash_stable(__hcx, __hasher);
            }
            Generator(def_id, substs, m) => {
                def_id.hash_stable(__hcx, __hasher);
                substs.hash_stable(__hcx, __hasher);
                m.hash_stable(__hcx, __hasher);
            }
            GeneratorWitness(b) => {
                b.hash_stable(__hcx, __hasher);
            }
            Never => {}
            Tuple(substs) => {
                substs.hash_stable(__hcx, __hasher);
            }
            Projection(p) => {
                p.hash_stable(__hcx, __hasher);
            }
            Opaque(def_id, substs) => {
                def_id.hash_stable(__hcx, __hasher);
                substs.hash_stable(__hcx, __hasher);
            }
            Param(p) => {
                p.hash_stable(__hcx, __hasher);
            }
            Bound(d, b) => {
                d.hash_stable(__hcx, __hasher);
                b.hash_stable(__hcx, __hasher);
            }
            Placeholder(p) => {
                p.hash_stable(__hcx, __hasher);
            }
            Infer(i) => {
                i.hash_stable(__hcx, __hasher);
            }
            Error(d) => {
                d.hash_stable(__hcx, __hasher);
            }
        }
    }
}
