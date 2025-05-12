trait Foo {
    const ASSOC: [u8];
}

fn bar<T: Foo>() {
    let a = [T::ASSOC; 2];
    //~^ ERROR: the size for values of type `[u8]` cannot be known at compilation time
}

fn main() {}
