#![crate_name = "foo"]

// @has foo/macro.bar.html
// @has foo/macro.bar!.html
// @!has foo/bar.m.html
#[macro_export]
macro_rules! bar {
    () => {}
}

// @has foo/struct.Bar.html
// @!has foo/Bar.t.html
pub struct Bar;
