#![crate_name = "foo"]

// Test to ensure identically named methods in the sidebar get a disambiguator

//@ has foo/struct.C.html
pub struct A;
pub struct B;
pub struct C<T>(T);

impl C<A> {
    //@ has - "//a[@href='#method.dupe']" "dupe (C<A>)"
    pub fn dupe(self) {}
    //@ has - "//a[@href='#method.uniq1']" "uniq1"
    //@ !has - "//a[@href='#method.uniq1']" "uniq1 (C<A>)"
    pub fn uniq1(self) {}
}

impl C<B> {
    //@ has - "//a[@href='#method.dupe-1']" "dupe (C<B>)"
    pub fn dupe(self) {}
    //@ has - "//a[@href='#method.uniq2']" "uniq2"
    //@ !has - "//a[@href='#method.uniq2']" "uniq2 (C<B>)"
    pub fn uniq2(self) {}
}
