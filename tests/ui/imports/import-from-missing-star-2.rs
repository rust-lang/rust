//@revisions: edition2015 edition2024
//@[edition2015] edition:2015
//@[edition2024] edition:2024
mod foo {
//[edition2015]~^ HELP you might be missing a crate named `spam`, add it to your project and import it in your code
    use spam::*; //~ ERROR unresolved import `spam` [E0432]
    //[edition2024]~^ HELP you might be missing a crate named `spam`
}

fn main() {
    // Expect this to pass because the compiler knows there's a failed `*` import in `foo` that
    // might have caused it.
    foo::bar();
    // FIXME: these two should *fail* because they can't be fixed by fixing the glob import in `foo`
    ham(); // should error but doesn't
    eggs(); // should error but doesn't
}
