// check-pass

pub fn f() -> impl Sized {
    pub enum E {
        V(E),
    }

    unimplemented!()
}
