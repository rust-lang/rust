// Test that we can use ! as an associated type.

// check-pass

trait Foo {
    type Wow;

    fn smeg(&self) -> Self::Wow;
}

struct Blah;
impl Foo for Blah {
    type Wow = !;
    fn smeg(&self) -> ! {
        panic!("kapow!");
    }
}

fn main() {
    Blah.smeg();
}
