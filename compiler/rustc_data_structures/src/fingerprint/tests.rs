use super::*;

// Check that `combine_commutative` is order independent.
#[test]
fn combine_commutative_is_order_independent() {
    let a = Fingerprint::new(0xf6622fb349898b06, 0x70be9377b2f9c610);
    let b = Fingerprint::new(0xa9562bf5a2a5303c, 0x67d9b6c82034f13d);
    let c = Fingerprint::new(0x0d013a27811dbbc3, 0x9a3f7b3d9142ec43);
    let permutations = [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)];
    let f = a.combine_commutative(b).combine_commutative(c);
    for p in &permutations {
        assert_eq!(f, p.0.combine_commutative(p.1).combine_commutative(p.2));
    }
}
