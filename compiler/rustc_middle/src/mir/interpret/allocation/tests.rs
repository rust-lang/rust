use super::*;

#[test]
fn uninit_mask() {
    let mut mask = InitMask::new(Size::from_bytes(500), false);
    assert!(!mask.get(Size::from_bytes(499)));
    mask.set_range(alloc_range(Size::from_bytes(499), Size::from_bytes(1)), true);
    assert!(mask.get(Size::from_bytes(499)));
    mask.set_range((100..256).into(), true);
    for i in 0..100 {
        assert!(!mask.get(Size::from_bytes(i)), "{i} should not be set");
    }
    for i in 100..256 {
        assert!(mask.get(Size::from_bytes(i)), "{i} should be set");
    }
    for i in 256..499 {
        assert!(!mask.get(Size::from_bytes(i)), "{i} should not be set");
    }
}
