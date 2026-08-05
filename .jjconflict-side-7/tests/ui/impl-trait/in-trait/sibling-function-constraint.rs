// Checks that a sibling function (i.e. `foo`) cannot constrain
// an RPITIT from another function (`bar`).

trait Trait {
    fn foo();

    fn bar() -> impl Sized;
}

impl Trait for () {
    fn foo() {
        let _: String = Self::bar();
        //~^ ERROR mismatched types
    }

    fn bar() -> impl Sized {
        loop {}
    }
}

fn main() {}
