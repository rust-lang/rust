pub struct S(pub ());

impl S {
    pub fn foo(&self) { }
}

pub trait T {
    fn bar(&self);
}

impl T for S {
    fn bar(&self) { }
}
