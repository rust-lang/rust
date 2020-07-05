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
