pub fn f() -> impl Sized {
    pub enum E {
        //~^ ERROR: recursive type
        V(E),
    }

    unimplemented!()
}
