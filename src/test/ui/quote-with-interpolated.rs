#![feature(quote)]
fn main() {
    macro_rules! foo {
        ($bar:expr)  => {
            quote_expr!(cx, $bar)
            //~^ ERROR quote! with interpolated token
            //~| ERROR failed to resolve: could not find `syntax` in `{{root}}`
            //~| ERROR failed to resolve: could not find `syntax` in `{{root}}`
            //~| ERROR cannot find value `cx` in this scope
            //~| ERROR cannot find function `new_parser_from_tts` in this scope
        }
    }
    foo!(bar);
}
