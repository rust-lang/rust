use std::num::NonZero;

use rustc_abi::Size;

use super::*;

fn id(value: u64) -> AllocId {
    AllocId(NonZero::new(value).unwrap())
}

#[test]
fn names_follow_identity_not_internal_numbers() {
    for (a, b) in [(100, 20), (20, 100)] {
        let ids = RefCell::default();
        let names = AllocationNames(Some(&ids));
        assert_eq!(
            format!("{:?} {:?} {:?}", names.wrap(id(a)), names.wrap(id(b)), names.wrap(id(a))),
            "alloc1 alloc2 alloc1"
        );
        assert_eq!(format!("{:#?}", names.wrap(id(b))), "a2");
    }
}

#[test]
fn pointer_offsets_and_provenance_are_preserved() {
    let ids = RefCell::default();
    let names = AllocationNames(Some(&ids));
    let provenance = CtfeProvenance::from(id(100)).as_immutable();
    let first = Pointer::new(provenance, Size::from_bytes(4));
    let second = Pointer::new(provenance, Size::from_bytes(8));
    assert_eq!(format!("{:?}", names.wrap(first)), "alloc1+0x4<imm>");
    assert_eq!(format!("{:#?}", names.wrap(second)), "a1+0x8<imm>");
    let other = Pointer::new(CtfeProvenance::from(id(20)), Size::ZERO);
    assert_eq!(format!("{:?}", names.wrap(other)), "alloc2");
}

#[test]
fn indirect_constants_keep_their_offset() {
    let ids = RefCell::default();
    let names = AllocationNames(Some(&ids));
    let value = ConstValue::Indirect { alloc_id: id(100), offset: Size::from_bytes(4) };
    assert_eq!(
        format!("{:?}", names.wrap(&value)),
        "Indirect { alloc_id: alloc1, offset: Size(4 bytes) }"
    );
}
