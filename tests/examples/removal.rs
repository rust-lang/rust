pub mod old {
    pub struct Abc;

    pub struct Bce;

    pub mod a {

    }

    pub mod b {

    }
}

pub mod new {
    #[allow(dead_code)]
    struct Bce;

    mod b {

    }
}
