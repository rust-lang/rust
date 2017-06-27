pub mod old {
    pub fn abc() {}

    pub fn bcd() {}

    pub fn cde() {}

    pub fn def(_: u8) {}
}

pub mod new {
    pub fn abc() {}

    pub fn bcd(_: u8) {}

    pub fn cde() -> u16 { // TODO: maybe make this case TechnicallyBreaking
        0xcde
    }

    pub fn def() {}
}
