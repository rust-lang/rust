//! This is an unusual feature gate test, as it doesn't test the feature
//! gate, but the fact that not adding the feature gate will cause the
//! diagnostic to not emit the custom diagnostic message.
#[diagnostic::on_unmatch_args(note = "custom note")]
macro_rules! pair {
    //~^ NOTE when calling this macro
    ($ty:ty, $value:expr) => {};
    //~^ NOTE while trying to match `,`
}

fn main() {
    pair!(u8);
    //~^ ERROR unexpected end of macro invocation
    //~| NOTE missing tokens in macro arguments
}
