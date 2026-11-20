use std::num::NonZero;

use rustc_abi::Size;

use super::*;

fn id(value: u64) -> AllocId {
    AllocId(NonZero::new(value).unwrap())
}

#[test]
fn names_follow_identity_not_internal_numbers() {
    for (a, b) in [(100, 20), (20, 100)] {
        let names = AllocationNames::default();
        assert_eq!(
            format!("{:?} {:?} {:?}", names.id(id(a)), names.id(id(b)), names.id(id(a))),
            "alloc1 alloc2 alloc1"
        );
        assert_eq!(format!("{:#?}", names.id(id(b))), "a2");
    }
}

#[test]
fn files_have_independent_names() {
    let first = AllocationNames::default();
    let second = AllocationNames::default();
    assert_eq!(format!("{:?}", first.id(id(100))), "alloc1");
    assert_eq!(format!("{:?}", second.id(id(20))), "alloc1");
    assert_eq!(format!("{:?}", first.id(id(20))), "alloc2");
}

#[test]
fn ordinary_id_formatting_is_independent_of_mir_names() {
    let names = AllocationNames::default();
    assert_eq!(format!("{:?}", names.id(id(100))), "alloc1");
    assert_eq!(format!("{:?}", id(100)), "alloc100");
    assert_eq!(format!("{:?}", RawAllocIds.id(id(100))), "alloc100");
    assert_eq!(format!("{:?}", id(999)), "alloc999");
    assert_eq!(format!("{:?}", names.id(id(20))), "alloc2");
}

#[test]
fn pointer_offsets_and_provenance_are_preserved() {
    let names = AllocationNames::default();
    let provenance = CtfeProvenance::from(id(100)).as_immutable();
    let first = Pointer::new(provenance, Size::from_bytes(4));
    let second = Pointer::new(provenance, Size::from_bytes(8));
    assert_eq!(format!("{:?}", names.pointer(first)), "alloc1+0x4<imm>");
    assert_eq!(format!("{:#?}", names.pointer(second)), "a1+0x8<imm>");
    assert_eq!(format!("{:?}", first), "alloc100+0x4<imm>");
    let other = Pointer::new(CtfeProvenance::from(id(20)), Size::ZERO);
    assert_eq!(format!("{:?}", names.pointer(other)), "alloc2");
}

#[test]
fn raw_constant_fallback_keeps_the_existing_format() {
    let values = [
        ConstValue::ZeroSized,
        ConstValue::Slice { alloc_id: id(100), meta: 3 },
        ConstValue::Indirect { alloc_id: id(100), offset: Size::from_bytes(4) },
        ConstValue::Scalar(Scalar::from_pointer(
            Pointer::new(id(100).into(), Size::from_bytes(4)),
            &rustc_abi::TargetDataLayout::default(),
        )),
    ];
    for value in values {
        assert_eq!(format!("{:?}", RawAllocIds.wrap(&value)), format!("{value:?}"));
        assert_eq!(format!("{:#?}", RawAllocIds.wrap(&value)), format!("{value:#?}"));
    }
    let names = AllocationNames::default();
    assert_eq!(format!("{:?}", names.wrap(&values[1])), "Slice { alloc_id: alloc1, meta: 3 }");
    assert_eq!(
        format!("{:?}", names.wrap(&values[2])),
        "Indirect { alloc_id: alloc1, offset: Size(4 bytes) }"
    );
}
