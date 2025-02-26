// Test crate used to check the `--generate-macro-expansion` option.
//@ compile-flags: -Zunstable-options --generate-macro-expansion --generate-link-to-definition

#[macro_export]
macro_rules! bar {
    ($x:ident) => {{
        $x += 2;
        $x *= 2;
    }}
}

#[derive(Debug)]
pub struct Bar {
    a: String,
    b: u8,
}

fn y_f(_: &str, _: &str, _: &str) {}

fn foo() {
    let mut y = 0;
    bar!(y);
    println!("
    {y}
    ");
    let s = y_f("\
bla", stringify!(foo), stringify!(bar));
}
