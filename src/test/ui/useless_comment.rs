#![deny(unused_doc_comments)]

fn foo() {
    /// a //~ ERROR doc comment not used by rustdoc
    let x = 12;

    /// multi-line //~ doc comment not used by rustdoc
    /// doc comment
    /// that is unused
    match x {
        /// c //~ ERROR doc comment not used by rustdoc
        1 => {},
        _ => {}
    }

    /// foo //~ ERROR doc comment not used by rustdoc
    unsafe {}

    #[doc = "foo"] //~ ERROR doc comment not used by rustdoc
    #[doc = "bar"] //~ ERROR doc comment not used by rustdoc
    3;
}

fn main() {
    foo();
}
