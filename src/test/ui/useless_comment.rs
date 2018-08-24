#![deny(unused_doc_comments)]

fn foo() {
    /// a //~ ERROR doc comment not used by rustdoc
    let x = 12;

    /// b //~ doc comment not used by rustdoc
    match x {
        /// c //~ ERROR doc comment not used by rustdoc
        1 => {},
        _ => {}
    }

    /// foo //~ ERROR doc comment not used by rustdoc
    unsafe {}
}

fn main() {
    foo();
}
