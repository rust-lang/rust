// Regression test for <https://github.com/rust-lang/rust/issues/165684>.
// Borrowck used to ICE when reporting a borrow that outlives a local of an inline const block.

const fn foo(_: &impl Fn(&u32) -> &u32) {}

fn main() {
    const {
        let x = 22;
        foo(&|_| &x)
        //~^ ERROR `x` does not live long enough
    }
}
