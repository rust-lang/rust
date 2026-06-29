#![feature(diagnostic_opaque)]

#[diagnostic::opaque]
#[macro_export]
macro_rules! wrap {
    ($x:ident) => {{
        let x = blah::$x;
    }};
}
