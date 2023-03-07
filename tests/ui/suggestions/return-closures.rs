fn foo() {
    //~^ HELP try adding a return type
    |x: &i32| 1i32
    //~^ ERROR mismatched types
}

fn bar(i: impl Sized) {
    //~^ HELP a return type might be missing here
    || i
    //~^ ERROR mismatched types
}

fn main() {}
