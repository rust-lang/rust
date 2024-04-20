use super::*;

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
            foo: t
            int: ti

            crate::foo
            int: ti
        "#]],
    );
}
