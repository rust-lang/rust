// Test crate used to check the `--generate-macro-expansion` option.
//@ compile-flags: -Zunstable-options --generate-macro-expansion --generate-link-to-definition

mod other;

#[macro_export]
macro_rules! bar {
    ($x:ident) => {{
        $x += 2;
        $x *= 2;
    }}
}

macro_rules! bar2 {
    () => {
        fn foo2() -> impl std::fmt::Display {
            String::new()
        }
    }
}

macro_rules! bar3 {
    () => {
        fn foo3() {}
        fn foo4() -> String { String::new() }
    }
}

bar2!();
bar3!();

#[derive(Debug, PartialEq)]
pub struct Bar;

#[derive(Debug
)]
pub struct Bar2;

fn y_f(_: &str, _: &str, _: &str) {}

fn foo() {
    let mut y = 0;
    bar!(y);
    println!("
    {y}
    ");
    // comment
    println!("
    {y}
    ");
    let s = y_f("\
bla", stringify!(foo), stringify!(bar));

    // Macro from another file.
    other_macro!(y);
}
