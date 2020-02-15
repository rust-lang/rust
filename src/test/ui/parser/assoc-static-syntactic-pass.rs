// Syntactically, we do allow e.g., `static X: u8 = 0;` as an associated item.

// check-pass

fn main() {}

#[cfg(FALSE)]
impl S {
    static IA: u8 = 0;
    static IB: u8;
    default static IC: u8 = 0;
    pub(crate) default static ID: u8;
}

#[cfg(FALSE)]
trait T {
    static TA: u8 = 0;
    static TB: u8;
    default static TC: u8 = 0;
    pub(crate) default static TD: u8;
}

#[cfg(FALSE)]
impl T for S {
    static TA: u8 = 0;
    static TB: u8;
    default static TC: u8 = 0;
    pub default static TD: u8;
}
