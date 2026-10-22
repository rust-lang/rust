use super::{RegionOrderKey, swap_region_eq_operands};

#[test]
fn keyed_unkeyed_equality_orientation() {
    let placeholder: RegionOrderKey = (2, 1, 0);
    assert!(!swap_region_eq_operands(Some(placeholder), None));
    assert!(swap_region_eq_operands(None, Some(placeholder)));
}

#[test]
fn stable_key_order_is_independent_of_operand_direction() {
    let a: RegionOrderKey = (0, 0, 0);
    let b: RegionOrderKey = (2, 1, 0);

    assert!(!swap_region_eq_operands(Some(a), Some(b)));
    assert!(swap_region_eq_operands(Some(b), Some(a)));
    assert!(!swap_region_eq_operands(Some(a), Some(a)));
}

#[test]
fn unknown_pair_has_no_proven_order() {
    assert!(!swap_region_eq_operands(None, None));
}
