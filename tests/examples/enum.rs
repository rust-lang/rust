pub mod old {
    pub enum Abc {

    }

    pub enum Bcd {
        Abc,
    }

    pub enum Cde {
        Abc,
    }

    pub enum Def {
        Abc,
        Bcd,
    }
}

pub mod new {
    pub enum Abc {
        Abc,
    }

    pub enum Bcd {

    }

    pub enum Cde {
        Abc,
        Bcd,
    }

    pub enum Def {
        Abc,
    }
}
