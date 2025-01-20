//@ compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

pub enum Ty {
    Var,
}

//@ has 'src/foo/jump-to-def-assoc-items.rs.html'
//@ has - '//a[@href="#6"]' 'Self::Var'
impl Ty {
    fn f() {
        let _ = Self::Var;
    }
}
