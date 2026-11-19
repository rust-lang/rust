//@ forbid-output: consider using `drop` function
// https://github.com/rust-lang/rust/issues/164044
macro_rules! call_drop {
    ($value:expr) => {
        $value.drop() //~ ERROR explicit use of destructor method
    };
}

fn main() {
    call_drop!(Vec::<u8>::new());
}
