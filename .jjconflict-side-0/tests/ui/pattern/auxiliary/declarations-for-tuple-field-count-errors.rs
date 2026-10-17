pub struct Z0;
pub struct Z1();

pub struct S(pub u8, pub u8, pub u8);
pub struct M(
    pub u8,
    pub u8,
    pub u8,
);

pub enum E1 { Z0, Z1(), S(u8, u8, u8) }

pub enum E2 {
    S(u8, u8, u8),
    M(
        u8,
        u8,
        u8,
    ),
}
