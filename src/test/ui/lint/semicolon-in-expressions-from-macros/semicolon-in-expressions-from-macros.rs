// check-pass
// edition:2018
#![warn(semicolon_in_expressions_from_macros)]

#[allow(dead_code)]
macro_rules! foo {
    ($val:ident) => {
        true; //~ WARN trailing
              //~| WARN this was previously
              //~| WARN trailing
              //~| WARN this was previously
    }
}

#[allow(semicolon_in_expressions_from_macros)]
async fn bar() {
    foo!(first);
}

fn main() {
    // This `allow` doesn't work
    #[allow(semicolon_in_expressions_from_macros)]
    let _ = {
        foo!(first)
    };

    // This 'allow' doesn't work either
    #[allow(semicolon_in_expressions_from_macros)]
    let _ = foo!(second);

    // But this 'allow' does
    #[allow(semicolon_in_expressions_from_macros)]
    fn inner() {
        let _ = foo!(third);
    }
}
