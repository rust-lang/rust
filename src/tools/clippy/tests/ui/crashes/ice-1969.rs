//@ check-pass

// Test for https://github.com/rust-lang/rust-clippy/issues/1969

fn main() {}

pub trait Convert {
    type Action: From<*const f64>;

    fn convert(val: *const f64) -> Self::Action {
        val.into()
    }
}
