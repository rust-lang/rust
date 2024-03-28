// aux-build:format_in_declarative_macro.rs

extern crate format_in_declarative_macro;
macro_rules! by_example {
    () => {
        fn by_example_fn(test: impl ::std::fmt::Display) {
            println!(test);
            //~^ERROR format argument must be a string literal
            //~|HELP you might be missing a string literal to format with
        }
    };
}

by_example!();
format_in_declarative_macro::external_decl!();
//~^ERROR format argument must be a string literal
//~|HELP you might be missing a string literal to format with
fn main() {}
