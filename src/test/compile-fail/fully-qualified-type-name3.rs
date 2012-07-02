// Test that we use fully-qualified type names in error messages.

type T1 = uint;
type T2 = int;

fn bar(x: T1) -> T2 {
    ret x;
    //~^ ERROR mismatched types: expected `T2` but found `T1`
}

fn main() {
}
