//@ check-pass
// https://github.com/rust-lang/rust/issues/115377

use module::*;

mod module {
    pub enum B {}
    impl B {
        pub const ASSOC: u8 = 0;
    }
}

#[derive()]
pub enum B {}
impl B {
    pub const ASSOC: u16 = 0;
}

macro_rules! m {
    ($right:expr) => {
        $right
    };
}

fn main() {
    let a: u16 = {
        use self::*;
        B::ASSOC
    };
    let b: u16 = m!({
        use self::*;
        B::ASSOC
    });
}
