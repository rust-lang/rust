// Regression test for https://github.com/rust-lang/rust/issues/145367
mod m {
    struct Priv2;
}
fn main() {
    WithUse { one: m::Priv2 } //~ ERROR: cannot find struct, variant or union type `WithUse` in this scope
    //~^ ERROR: unit struct `Priv2` is private
}
