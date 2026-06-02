struct Test;

enum Foo {
    #[repr(u8)]
    //~^ ERROR `#[repr]` attribute cannot be used on
    Variant,
}

fn main() {}
