#[allow(dead_code)]
pub mod old {
    pub struct Abc<A> {
        field: A,
    }

    pub struct Def {
        field: u8,
    }

    pub struct Efg {
        pub field: u8,
    }
}

#[allow(dead_code)]
pub mod new {
    pub struct Abc<B> {
        field: B,
    }

    pub struct Def<A=u8> {
        field: A,
    }

    pub struct Efg {
        pub field: u16,
    }
}
