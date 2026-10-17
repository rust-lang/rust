use rustc_middle::ty::TyCtxt;
use rustc_span::{BytePos, Span};
use rustc_type_ir::region_constraint::{And, LeafRegionConstraint, Or};

#[test]
fn canonicalization_preserves_only_one_ambiguity() {
    let first = Span::with_root_ctxt(BytePos(1), BytePos(2));
    let second = Span::with_root_ctxt(BytePos(3), BytePos(4));

    let first = LeafRegionConstraint::Ambiguity::<TyCtxt<'_>, _>(first);
    let second = LeafRegionConstraint::Ambiguity::<TyCtxt<'_>, _>(second);

    let c = And::new([first.clone(), second.clone()]);
    assert_eq!(c.0.len(), 1);

    let c = Or::new([And::new([first]), And::new([second])]);
    assert_eq!(c.0.len(), 1);
}
