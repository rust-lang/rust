struct Test;

enum Foo {
    #[repr(u8)]
    //~^ ERROR attribute should be applied to an enum
    Variant,
}

fn main() {}
