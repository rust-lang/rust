// aux-build:raw-foreign.rs
// build-aux-docs

extern crate raw_foreign;

// has 'raw_idents/trait.MyTrait.html' '//a[@href="../raw_foreign/try/trait.trait.html"]' 'trait'
// has 'raw_idents/trait.MyTrait.html' '//a[@href="../raw_foreign/try/struct.struct.html"]' 'struct'
pub trait MyTrait {
    fn foo<T>() where T: raw_foreign::r#try::r#trait {}
}

impl MyTrait for raw_foreign::r#try::r#struct {
    fn foo<T>() where T: raw_foreign::r#try::r#trait {}
}
