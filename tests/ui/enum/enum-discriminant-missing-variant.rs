//! regression test for issue <https://github.com/rust-lang/rust/issues/23217>
pub enum SomeEnum {
    B = SomeEnum::A, //~ ERROR no variant or associated item named `A` found
}

fn main() {}
