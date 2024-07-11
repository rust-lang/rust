// https://github.com/rust-lang/rust/issues/82465
#![crate_name = "foo"]

use std::convert::AsRef;
pub struct Local;

//@ has foo/struct.Local.html '//h3[@class="code-header"]' 'impl AsRef<str> for Local'
impl AsRef<str> for Local {
    fn as_ref(&self) -> &str {
        todo!()
    }
}

//@ has - '//h3[@class="code-header"]' 'impl AsRef<Local> for str'
impl AsRef<Local> for str {
    fn as_ref(&self) -> &Local {
        todo!()
    }
}
