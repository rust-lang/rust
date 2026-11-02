//@ check-pass
//@ run-rustfix

#![feature(f16, f128)]

extern crate core;

fn main() {
    let x = 5f16;
    let _ = x == f16::NAN;
    //~^ WARN incorrect NaN comparison
    let _ = x != f16::NAN;
    //~^ WARN incorrect NaN comparison

    let x = 5f32;
    let _ = x == f32::NAN;
    //~^ WARN incorrect NaN comparison
    let _ = x != f32::NAN;
    //~^ WARN incorrect NaN comparison

    let x = 5f64;
    let _ = x == f64::NAN;
    //~^ WARN incorrect NaN comparison
    let _ = x != f64::NAN;
    //~^ WARN incorrect NaN comparison

    let x = 5f128;
    let _ = x == f128::NAN;
    //~^ WARN incorrect NaN comparison
    let _ = x != f128::NAN;
    //~^ WARN incorrect NaN comparison

    let b = &2.3f32;
    if b != &f32::NAN {}
    //~^ WARN incorrect NaN comparison

    let b = &2.3f32;
    if b != { &f32::NAN } {}
    //~^ WARN incorrect NaN comparison

    let _ =
        b != {
    //~^ WARN incorrect NaN comparison
            &f32::NAN
        };

    // Issue #164143: an unsuffixed float is constrained by the `NAN` operand.
    // Suggesting `.is_nan()` drops that constraint and does not compile.
    let b = &3.3;
    if b != &f64::NAN {}
    //~^ WARN incorrect NaN comparison

    let mut m = &3.3;
    if m != &f64::NAN {
        //~^ WARN incorrect NaN comparison
        m = &1.0;
    }
    let _ = m;

    let unary = &(-3.3);
    if unary != &f64::NAN {}
    //~^ WARN incorrect NaN comparison

    let block = &{ 3.3 };
    if block != &f64::NAN {}
    //~^ WARN incorrect NaN comparison

    let (tuple,) = (&3.3,);
    if tuple != &f64::NAN {}
    //~^ WARN incorrect NaN comparison

    let arithmetic = &(3.3 + 0.0);
    if arithmetic != &f64::NAN {}
    //~^ WARN incorrect NaN comparison

    let x = 1.0f64;
    if &x != &f64::NAN {}
    //~^ WARN incorrect NaN comparison
    let p = &x;
    let _ = *p == f64::NAN;
    //~^ WARN incorrect NaN comparison
    let _ = -x == f64::NAN;
    //~^ WARN incorrect NaN comparison

    shadowed::check();
    nested_operands();
    macro_paths();
    method_resolution::check();

    let _ = 3.3 == f64::NAN;
    //~^ WARN incorrect NaN comparison

    let _ = &3.3 != &f64::NAN;
    //~^ WARN incorrect NaN comparison

    let suffixed = 3.3f64;
    let copied = suffixed;
    let _ = copied != f64::NAN;
    //~^ WARN incorrect NaN comparison

    let ascribed: f64 = 3.3;
    let _ = ascribed != f64::NAN;
    //~^ WARN incorrect NaN comparison

    parameter(1.0);
    parameter_ref(&1.0);
    let _ = |x: f64| x != f64::NAN; //~ WARN incorrect NaN comparison

    #[allow(unused_macros)]
    macro_rules! nan { () => { f32::NAN }; }
    macro_rules! number { () => { 5f32 }; }

    let _ = nan!() == number!();
    //~^ WARN incorrect NaN comparison
    let _ = number!() != nan!();
    //~^ WARN incorrect NaN comparison
}

fn parameter(x: f64) {
    let _ = x == f64::NAN;
    //~^ WARN incorrect NaN comparison
}

fn parameter_ref(x: &f32) {
    if x != &f32::NAN {}
    //~^ WARN incorrect NaN comparison
}

fn nested_operands() {
    let x = 1.0f64;
    let _ = &x as &f64 == &f64::NAN;
    //~^ WARN incorrect NaN comparison
    let _ = { &{ 3.3 } } != &f64::NAN;
    //~^ WARN incorrect NaN comparison
    let _ = &&&3.3 == &&&f64::NAN;
    //~^ WARN incorrect NaN comparison
    let b = &&&3.3;
    let _ = &&&f64::NAN != b;
    //~^ WARN incorrect NaN comparison
    let b = &mut 3.3;
    let _ = b != &f64::NAN;
    //~^ WARN incorrect NaN comparison
    let b = &3.3;
    let _ = &(<f64>::NAN) == b;
    //~^ WARN incorrect NaN comparison
    let b = &3.3;
    let _ = b != &{ f64::NAN };
    //~^ WARN incorrect NaN comparison
    let b = &&3.3;
    let _ = &{ &{ f64::NAN } } == b;
    //~^ WARN incorrect NaN comparison

    let b = &(-3.3);
    let _ = b != &f16::NAN;
    //~^ WARN incorrect NaN comparison
    let b = &(3.3 + 0.0);
    let _ = b == &f128::NAN;
    //~^ WARN incorrect NaN comparison

    macro_rules! borrowed { () => { &3.3 }; }
    let _ = borrowed!() != &f64::NAN;
    //~^ WARN incorrect NaN comparison
}

pub type MacroFloat = f64;

fn macro_paths() {
    #[allow(unused_macros)]
    macro_rules! nan { () => { $crate::MacroFloat::NAN }; }
    let b = &3.3;
    let _ = b != &nan!();
    //~^ WARN incorrect NaN comparison
}

mod shadowed {
    #[allow(dead_code, non_camel_case_types)]
    struct f64;
    use ::core::primitive::f64 as Float;

    pub fn check() {
        let b = &3.3;
        if b != &::core::primitive::f64::NAN {}
        //~^ WARN incorrect NaN comparison
        let c = &4.4;
        if c != &Float::NAN {}
        //~^ WARN incorrect NaN comparison
        let d = &5.5;
        if d != &Float::r#NAN {}
        //~^ WARN incorrect NaN comparison
    }
}

mod method_resolution {
    mod f64 {
        pub fn is_nan(_: ::core::primitive::f64) {}
    }

    trait IsNan {
        fn is_nan(self);
    }

    impl IsNan for &f64 {
        fn is_nan(self) {}
    }

    pub fn check() {
        let _: () = f64::is_nan(5.0);
        let _: () = (&5f64).is_nan();

        let x = 5f64;
        let _: bool = x == f64::NAN;
        //~^ WARN incorrect NaN comparison

        let x = &5f64;
        let _: bool = x == &f64::NAN;
        //~^ WARN incorrect NaN comparison
    }
}
