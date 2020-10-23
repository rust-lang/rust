// check-pass

fn main() {

    #[allow(unused_doc_comments)]
    /// A doc comment
    fn bar() {}

    /// Another doc comment
    #[allow(unused_doc_comments)]
    struct Foo {}
}
