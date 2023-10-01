// Regression test for #114918
// Test that a const generic enclosed in a block within the return type
// of a trait method produces a type mismatch error instead of triggering
// a const eval cycle

#[allow(unused_braces)]
trait Trait {
    fn func<const N: u32>() -> [ (); { () }] { //~ ERROR mismatched types
        N
    }
}

fn main() {}
