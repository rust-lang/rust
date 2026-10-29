// This test ensures that we have the expected number of line generated.

#![crate_name = "foo"]

//@ has 'src/foo/source-line-numbers.rs.html'
//@ count - '//a[@data-nosnippet]' 35
//@ has - '//a[@id="35"]' '35'

#[
macro_export
]
macro_rules! bar {
    ($x:ident) => {{
        $x += 2;
        $x *= 2;
    }}
}

/*
multi line
comment
*/
fn x(_: u8, _: u8) {}

fn foo() {
    let mut y = 0;
    bar!(y);
    println!("
    {y}
    ");
    x(
      1,
      2,
    );
}
