// Most traits cannot be derived for unions.

#[derive(
    PartialEq, //~ ERROR this trait cannot be derived for unions
    PartialOrd, //~ ERROR this trait cannot be derived for unions
    Ord, //~ ERROR this trait cannot be derived for unions
    Hash, //~ ERROR this trait cannot be derived for unions
    Default, //~ ERROR this trait cannot be derived for unions
    Debug, //~ ERROR this trait cannot be derived for unions
)]
union U {
    a: u8,
    b: u16,
}

fn main() {}
