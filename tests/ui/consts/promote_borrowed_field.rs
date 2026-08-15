//@ run-pass

// From https://github.com/rust-lang/rust/issues/65727

const _: &i32 = {
    let x = &(5, false).0;
    x
};

fn main() {
    let _: &'static i32 = &(5, false).0;
}
