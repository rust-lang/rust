use super::*;

#[test]
fn kw_path_renames() {
    check(
        r#"
macro_rules! m {
    () => {
        pub use $crate as dollar_crate;
        pub use $crate::{self as self_dollar_crate};
    };
}

pub use self as this;
pub use crate as krate;

pub use crate::{self as self_krate};
m!();

mod foo {
    pub use super as zuper;
    pub use super::{self as self_zuper};
}
"#,
        expect![[r#"
            crate
            - dollar_crate : type (import)
            - foo : type
            - krate : type (import)
            - self_dollar_crate : type (import)
            - self_krate : type (import)
            - this : type (import)
            - (legacy) m : macro!

            crate::foo
            - self_zuper : type (import)
            - zuper : type (import)
            - (legacy) m : macro!
        "#]],
    );
}

#[test]
fn primitive_reexport() {
    check(
        r#"
//- /lib.rs
mod foo;
use foo::int;

//- /foo.rs
pub use i32 as int;
"#,
        expect![[r#"
            crate
            - foo : type
            - int : type (import)

            crate::foo
            - int : type (import)
        "#]],
    );
}
