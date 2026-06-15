// This test ensures that the `foo` decl macro is present in the module sidebar.

#![feature(decl_macro)]
#![crate_name = "foo"]

//@has 'foo/bar/index.html'
//@has - '//*[@id="rustdoc-modnav"]/ul[@class="block macro"]//a[@href="../macro.foo.html"]' 'foo'

pub macro foo {
    () => { "bar" }
}

/// docs
pub mod bar {
}
