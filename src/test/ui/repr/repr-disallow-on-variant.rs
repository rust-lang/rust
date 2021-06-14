struct Test;

enum Foo {
    #[repr(u8)]
    //~^ ERROR attribute should be applied to a struct, enum, or union
    Variant,
}

fn main() {}
