// This test makes sure that expended items with derives don't interfear with lint expectations.
//
// See <https://github.com/rust-lang/rust/issues/153036> for some context.

//@ check-pass

#[derive(Clone, Debug)]
#[expect(unused)]
pub struct LoggingArgs {
    #[cfg(false)]
    x: i32,
    y: i32,
}

fn main() {}
