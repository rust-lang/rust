// Prior to 1.64.0, '_ in dyn trait bounds was not counted as a new lifetime.
// Changes made in 1.64.0 broke this. This is a regression test for the fix.

// check-pass

trait Trait {}

// Case 1: Anonymous/elided lifetime on return type should refer to the `&` in `&(dyn ...)`.
fn a1(_: &(dyn Trait + '_)) -> &str { loop {} }
fn a2(_: &(dyn Trait + '_)) -> &'_ str { loop {} }
fn a3(_: &'_ (dyn Trait + '_)) -> &'_ str { loop {} }
fn a4(_: &(Trait + '_)) -> &str { loop {} } //~ WARNING deprecated
fn a_closure() {
    let _ = |_: &(dyn Trait + '_)| -> &str { loop {} };
    let _ = |_: &(Trait + '_)| -> &str { loop {} }; //~ WARNING deprecated
}

// Case 2: Elided lifetime on return type should refer to the `&` in `&i32`.
fn b1(_: &i32, _: Box<dyn Trait + '_>) -> &str { loop {} }
fn b2(_: &i32, _: Box<Trait + '_>) -> & str { loop {} } //~ WARNING deprecated
fn b3(_: Box<dyn Trait + '_>, _: &i32) -> &str { loop {} }
fn b4(_: Box<Trait + '_>, _: &i32) -> & str { loop {} } //~ WARNING deprecated
fn b_closure() {
    let _ = |_: &i32, _: Box<dyn Trait + '_>| -> &str { loop {} };
    let _ = |_: &i32, _: Box<Trait + '_>| -> &str { loop {} };
    let _ = |_: Box<dyn Trait + '_>, _: &i32| -> &str { loop {} };
    let _ = |_: Box<Trait + '_>, _: &i32| -> &str { loop {} };
}

// Case 3: Anonymous lifetime bound in trait object should not be resolved against previous ones.
fn c1(_: &i32, _: &i32, _: Box<dyn Trait + '_>) {}
fn c2(_: &i32, _: &i32, _: Box<Trait + '_>) {}
fn c_closure() {
    let _ = |_: &i32, _: &i32, _: Box<dyn Trait + '_>| loop {};
    let _ = |_: &i32, _: &i32, _: Box<Trait + '_>| loop {};
}

fn main() {}
