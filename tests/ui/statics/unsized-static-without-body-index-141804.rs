// Regression test for <https://github.com/rust-lang/rust/issues/141804>.
// Indexing an unsized `static` declared without a body used to ICE in const eval
// with "primitive read not possible for type".

fn foo() {
    static symbol: [u32];
    //~^ ERROR free static item without body
    //~| ERROR the size for values of type `[u32]` cannot be known at compilation time
    symbol[0];
}

fn main() {}
