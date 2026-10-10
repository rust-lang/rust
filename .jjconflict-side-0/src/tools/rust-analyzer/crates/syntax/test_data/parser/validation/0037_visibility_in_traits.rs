impl T for () {
    fn foo() {}
    pub fn bar() {}
    pub(crate) type Baz = ();
    pub(crate) const C: i32 = 92;
}
