struct Example {
    foo: usize
}

struct Example2(u32);

const EXAMPLE: Example = Example { foo: 42 };
const EXAMPLE_2: Example2 = Example2(42);

struct Wow<const N: usize> {
    field: [u8; N]
}


fn a() {
    let _a: Wow<EXAMPLE.foo> = Wow::new();
    //~^ ERROR expected one of
}

fn main() {
    let _b: Wow<EXAMPLE_2.0> = Wow::new();
    //~^ ERROR expected one of
}
