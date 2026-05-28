//! Regression test for https://github.com/rust-lang/rust/issues/16149

extern "C" {
    static externalValue: isize;
}

fn main() {
    let boolValue = match 42 {
        externalValue => true,
        //~^ ERROR match bindings cannot shadow statics
        _ => false,
    };
}
