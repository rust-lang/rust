// build-pass (FIXME(62277): could be check-pass?)

#![deny(private_doc_tests)]

mod foo {
    /**
    Does nothing, returns `()`

    yadda-yadda-yadda
    */
    fn foo() {}
}
