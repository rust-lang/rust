// Check that nested items have their visibility and `default`nesses in the right order.

//@ pp-exact

fn main() {}

#[cfg(FALSE)]
extern "C" {
    static X: u8;
    type X;
    fn foo();
    pub static X: u8;
    pub type X;
    pub fn foo();
}

#[cfg(FALSE)]
trait T {
    const X: u8;
    type X;
    fn foo();
    default const X: u8;
    default type X;
    default fn foo();
    pub const X: u8;
    pub type X;
    pub fn foo();
    pub default const X: u8;
    pub default type X;
    pub default fn foo();
}

#[cfg(FALSE)]
impl T for S {
    const X: u8;
    type X;
    fn foo();
    default const X: u8;
    default type X;
    default fn foo();
    pub const X: u8;
    pub type X;
    pub fn foo();
    pub default const X: u8;
    pub default type X;
    pub default fn foo();
}
