pub mod old {
    #[allow(dead_code)]
    struct Bcd;

    mod b {

    }
}

pub mod new {
    pub struct Abc;

    pub struct Bcd;

    pub mod a {

    }

    pub mod b {
        #[allow(dead_code)]
        pub struct Cde;
    }
}
