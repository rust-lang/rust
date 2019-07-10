use libm::*;

#[test]
fn fma_segfault() {
    // These two inputs cause fma to segfault on release due to overflow:
    assert_eq!(
        fma(
            -0.0000000000000002220446049250313,
            -0.0000000000000002220446049250313,
            -0.0000000000000002220446049250313
        ),
        -0.00000000000000022204460492503126,
    );

    assert_eq!(
        fma(-0.992, -0.992, -0.992),
        -0.00793599999988632,
    );
}
