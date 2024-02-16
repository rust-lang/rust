//@ check-pass
//@ edition:2018
#![feature(stmt_expr_attributes)]
#![warn(semicolon_in_expressions_from_macros)]

#[allow(dead_code)]
macro_rules! foo {
    ($val:ident) => {
        true; //~  WARN trailing semicolon in macro
              //~| WARN this was previously accepted
              //~| WARN trailing semicolon in macro
              //~| WARN this was previously accepted
              //~| WARN trailing semicolon in macro
              //~| WARN this was previously accepted
    }
}

#[allow(semicolon_in_expressions_from_macros)]
async fn bar() {
    foo!(first);
}

fn main() {
    #[allow(semicolon_in_expressions_from_macros)]
    let _ = {
        foo!(first)
    };

    #[allow(semicolon_in_expressions_from_macros)]
    let _ = foo!(second);

    #[allow(semicolon_in_expressions_from_macros)]
    fn inner() {
        let _ = foo!(third);
    }

    #[allow(semicolon_in_expressions_from_macros)]
    async {
        let _ = foo!(fourth);
    };

    let _ = {
        foo!(warn_in_block)
    };

    let _ = foo!(warn_in_expr);

    // This `#[allow]` does not work, since the attribute gets dropped
    // when we expand the macro
    let _ = #[allow(semicolon_in_expressions_from_macros)] foo!(allow_does_not_work);
}
