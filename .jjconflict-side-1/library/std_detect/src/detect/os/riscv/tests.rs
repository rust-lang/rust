use super::*;

#[test]
fn simple_direct() {
    let mut value = cache::Initializer::default();
    value.set(Feature::f as u32);
    // F (and other extensions with CSRs) -> Zicsr
    assert!(imply_features(value).test(Feature::zicsr as u32));
}

#[test]
fn simple_indirect() {
    let mut value = cache::Initializer::default();
    value.set(Feature::q as u32);
    // Q -> D, D -> F, F -> Zicsr
    assert!(imply_features(value).test(Feature::zicsr as u32));
}

#[test]
fn complex_zcd() {
    let mut value = cache::Initializer::default();
    // C & D -> Zcd
    value.set(Feature::c as u32);
    assert!(!imply_features(value).test(Feature::zcd as u32));
    value.set(Feature::d as u32);
    assert!(imply_features(value).test(Feature::zcd as u32));
}

#[test]
fn group_simple_forward() {
    let mut value = cache::Initializer::default();
    // A -> Zalrsc & Zaamo (forward implication)
    value.set(Feature::a as u32);
    let value = imply_features(value);
    assert!(value.test(Feature::zalrsc as u32));
    assert!(value.test(Feature::zaamo as u32));
}

#[test]
fn group_simple_backward() {
    let mut value = cache::Initializer::default();
    // Zalrsc & Zaamo -> A (reverse implication)
    value.set(Feature::zalrsc as u32);
    value.set(Feature::zaamo as u32);
    assert!(imply_features(value).test(Feature::a as u32));
}

#[test]
fn group_complex_convergence() {
    let mut value = cache::Initializer::default();
    // Needs 3 iterations to converge
    // (and 4th iteration for convergence checking):
    // 1.  [Zvksc] -> Zvks & Zvbc
    // 2.  Zvks -> Zvksed & Zvksh & Zvkb & Zvkt
    // 3a. [Zvkned] & [Zvknhb] & [Zvkb] & Zvkt -> {Zvkn}
    // 3b. Zvkn & Zvbc -> {Zvknc}
    value.set(Feature::zvksc as u32);
    value.set(Feature::zvkned as u32);
    value.set(Feature::zvknhb as u32);
    value.set(Feature::zvkb as u32);
    let value = imply_features(value);
    assert!(value.test(Feature::zvkn as u32));
    assert!(value.test(Feature::zvknc as u32));
}
