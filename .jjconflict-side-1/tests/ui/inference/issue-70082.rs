fn main() {
    // this closure is fine, and should not get any error annotations
    let em = |v: f64| -> f64 { v };

    let x: f64 = em(1i16.into());

    let y: f64 = 0.01f64 * 1i16.into();
    //~^ ERROR type annotations needed
    //~| HELP try using a fully qualified path
}
