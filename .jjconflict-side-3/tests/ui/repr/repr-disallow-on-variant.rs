struct Test;

enum Foo {
    #[repr(u8)]
    //~^ ERROR attribute cannot be used on
    Variant,
}

fn main() {}
