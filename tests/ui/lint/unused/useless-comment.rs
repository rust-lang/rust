#![feature(stmt_expr_attributes)]

#![deny(unused_doc_comments)]

macro_rules! mac {
    () => {}
}

/// foo //~ ERROR unused doc comment
mac!();

/// a //~ ERROR unused doc comment
#[doc(test(attr(allow(dead_code))))] //~ ERROR unused doc comment
unsafe extern "C" { }

fn foo() {
    /// a //~ ERROR unused doc comment
    #[doc(test(attr(allow(dead_code))))] //~ ERROR unused doc comment
    let x = 12;

    /// multi-line //~ ERROR unused doc comment
    /// doc comment
    /// that is unused
    match x {
        /// c //~ ERROR unused doc comment
        1 => {},
        #[doc(test(attr(allow(dead_code))))] //~ ERROR unused doc comment
        _ => {}
    }

    /// foo //~ ERROR unused doc comment
    unsafe {}

    #[doc = "foo"] //~ ERROR unused doc comment
    #[doc = "bar"] //~ ERROR unused doc comment
    3;

    /// bar //~ ERROR unused doc comment
    mac!();

    #[doc(test(attr(allow(dead_code))))] //~ ERROR unused doc comment
    let x = /** comment */ 47; //~ ERROR unused doc comment

    /// dox //~ ERROR unused doc comment
    {

    }
}

fn main() {
    foo();
}
