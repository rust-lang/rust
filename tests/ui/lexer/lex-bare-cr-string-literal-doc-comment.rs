// ignore-tidy-cr

/// doc comment with bare CR: ''
pub fn foo() {}
//~^^ ERROR: bare CR not allowed in doc-comment

/** block doc comment with bare CR: '' */
pub fn bar() {}
//~^^ ERROR: bare CR not allowed in block doc-comment

fn main() {
    //! doc comment with bare CR: ''
    //~^ ERROR: bare CR not allowed in doc-comment

    /*! block doc comment with bare CR: '' */
    //~^ ERROR: bare CR not allowed in block doc-comment

    // the following string literal has a bare CR in it
    let _s = "foobar"; //~ ERROR: bare CR not allowed in string

    // the following string literal has a bare CR in it
    let _s = r"barfoo"; //~ ERROR: bare CR not allowed in raw string

    // the following string literal has a bare CR in it
    let _s = "foo\bar"; //~ ERROR: unknown character escape: `\r`
}
