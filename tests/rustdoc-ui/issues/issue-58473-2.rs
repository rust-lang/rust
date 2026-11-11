//@ check-pass

#![deny(rustdoc::private_doc_tests)]

mod foo {
    /**
    Does nothing, returns `()`

    yadda-yadda-yadda
    */
    fn foo() {}
}
