#[test]
fn f32_ref() {
    let x: f32 = -0.0;
    let still_x: f32 = [x].iter().sum();
    assert_eq!(1. / x, 1. / still_x)
}

#[test]
fn f32_own() {
    let x: f32 = -0.0;
    let still_x: f32 = [x].into_iter().sum();
    assert_eq!(1. / x, 1. / still_x)
}

#[test]
fn f64_ref() {
    let x: f64 = -0.0;
    let still_x: f64 = [x].iter().sum();
    assert_eq!(1. / x, 1. / still_x)
}

#[test]
fn f64_own() {
    let x: f64 = -0.0;
    let still_x: f64 = [x].into_iter().sum();
    assert_eq!(1. / x, 1. / still_x)
}
