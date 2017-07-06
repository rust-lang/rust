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

    pub enum Efg {
        Abc,
        Bcd(u8),
        Cde,
        Def { f: u8 },
        Efg(u8),
        Fgh { f: u8 },
        Ghi { f: u8 },
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

    pub enum Efg {
        Abc(u8),
        Bcd,
        Cde { f: u8 },
        Def,
        Efg { f: u8 },
        Fgh { f: u16 },
        Ghi { g: u8 },
    }
}
