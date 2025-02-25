extern "unadjusted" fn foo() {
//~^ ERROR: "unadjusted" ABI is an implementation detail and perma-unstable
}

fn main() {
    foo();
}
