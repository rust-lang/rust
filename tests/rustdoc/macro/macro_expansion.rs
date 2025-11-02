// This test checks that patterns and statements are also getting expanded.

//@ compile-flags: -Zunstable-options --generate-macro-expansion

#![crate_name = "foo"]

//@ has 'src/foo/macro_expansion.rs.html'
//@ count - '//span[@class="expansion"]' 2

macro_rules! pat {
    ($x:literal) => {
        Some($x)
    }
}

macro_rules! stmt {
    ($x:expr) => {{
        let _ = $x;
    }}
}

fn bar() {
    match Some("hello") {
        pat!("blolb") => {}
        _ => {}
    }
    stmt!(1)
}
