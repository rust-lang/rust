// Test that we do not attempt to create dyn-incompatible trait objects in const eval.

trait Qux {
    fn bar();
}

static FOO: &(dyn Qux + Sync) = "desc";
//~^ the trait `Qux` is not dyn compatible
//~| the trait `Qux` is not dyn compatible
//~| the trait `Qux` is not dyn compatible

fn main() {}
