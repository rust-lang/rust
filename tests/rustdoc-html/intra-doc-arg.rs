#![crate_name = "foo"]
#![feature(intra_doc_arg)]

/// **[arg@x1]**
//@ !has foo/index.html '//strong a' 'x1'
//@ has foo/index.html '//strong' 'x1'
pub fn a(x1: ()) {}

pub struct X;

impl X {
    /// **[arg@x2]**
    //@ !has foo/struct.X.html '//strong a' 'x2'
    //@ has foo/struct.X.html '//strong' 'x2'
    pub fn a(x2: ()) {}
}

pub trait T {
    /// **[`arg@x3`]**
    //@ !has foo/trait.T.html '//strong a' 'x3'
    //@ has foo/trait.T.html '//strong' 'x3'
    fn a(x3: ()) {}

    /// **[arg@x4]**
    //@ !has foo/trait.T.html '//strong a' 'x4'
    //@ has foo/trait.T.html '//strong' 'x4'
    fn b(x4: ());
}

extern "C" {
    /// **[`arg@x5`]**
    //@ !has foo/index.html '//strong a' 'x5'
    //@ has foo/index.html '//strong' 'x5'
    pub fn x(x5: ());
}
