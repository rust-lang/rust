//! Regression test for <https://github.com/rust-lang/rust/issues/161915>.
//! Mismatched RPITIT projections should produce E0308 instead of an ICE.

trait Parameter {
    fn create() -> impl Form;
}

trait Form {}

fn forms_at_phase<T: Parameter, U: Parameter>() -> impl Form {
    match true {
        true => T::create(),
        false => U::create(), //~ ERROR `match` arms have incompatible types
    }
}

fn main() {}
