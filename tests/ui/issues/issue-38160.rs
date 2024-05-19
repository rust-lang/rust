//@ check-pass

trait MyTrait {
    const MY_CONST: &'static str;
}

macro_rules! my_macro {
    () => {
        struct MyStruct;

        impl MyTrait for MyStruct {
            const MY_CONST: &'static str = stringify!(abc);
        }
    }
}

my_macro!();

fn main() {}
