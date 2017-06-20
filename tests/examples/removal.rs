pub mod old {
    pub struct Abc;

    pub struct Bcd;

    pub mod a {

    }

    pub mod b {

    }
}

pub mod new {
    #[allow(dead_code)]
    struct Bcd;

    mod b {
        #[allow(dead_code)]
        pub struct Cde;
    }
}
