//! regression test for https://github.com/rust-lang/rust/issues/17001, https://github.com/rust-lang/rust/issues/21449, https://github.com/rust-lang/rust/issues/23189
mod foo {}

fn main() {
    let p = foo { x: () }; //~ ERROR expected struct, variant or union type, found module `foo`
}
