#![feature(quote)]
fn main() {
    macro_rules! foo {
        ($bar:expr)  => {
            quote_expr!(cx, $bar) //~ ERROR quote! with interpolated token
        }
    }
    foo!(bar);
}
