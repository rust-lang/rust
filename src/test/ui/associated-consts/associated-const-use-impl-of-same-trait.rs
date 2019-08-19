// run-pass

// The main purpose of this test is to ensure that different impls of the same
// trait can refer to each other without setting off the static recursion check
// (as long as there's no actual recursion).

trait Foo {
    const BAR: u32;
}

struct IsFoo1;

impl Foo for IsFoo1 {
    const BAR: u32 = 1;
}

struct IsFoo2;

impl Foo for IsFoo2 {
    const BAR: u32 = <IsFoo1 as Foo>::BAR;
}

fn main() {
    assert_eq!(<IsFoo1>::BAR, <IsFoo2 as Foo>::BAR);
}
