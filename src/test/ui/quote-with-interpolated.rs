#![feature(quote)]
fn main() {
    macro_rules! foo {
        ($bar:expr)  => {
            quote_expr!(cx, $bar)
            //~^ ERROR quote! with interpolated token
            //~| ERROR failed to resolve: maybe a missing `extern crate syntax;`?
            //~| ERROR failed to resolve: maybe a missing `extern crate syntax;`?
            //~| ERROR cannot find value `cx` in this scope
            //~| ERROR cannot find function `new_parser_from_tts` in this scope
        }
    }
    foo!(bar);
}
