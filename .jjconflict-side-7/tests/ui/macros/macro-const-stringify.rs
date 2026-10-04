//@ check-pass
//Tests to check that a macro generating an impl block with stringify for a trait associated const
//https://github.com/rust-lang/rust/issues/38160
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
