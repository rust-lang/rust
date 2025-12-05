#![feature(negative_impls)]

pub trait Tr {
    fn f();
    const C: u8;
    type T;
}
pub struct S {
    pub a: u8
}
struct Ts(pub u8);

pub impl Tr for S {  //~ ERROR visibility qualifiers are not permitted here
    pub fn f() {} //~ ERROR visibility qualifiers are not permitted here
    pub const C: u8 = 0; //~ ERROR visibility qualifiers are not permitted here
    pub type T = u8; //~ ERROR visibility qualifiers are not permitted here
}
pub impl S { //~ ERROR visibility qualifiers are not permitted here
    pub fn f() {}
    pub const C: u8 = 0;
    // pub type T = u8;
}
pub extern "C" { //~ ERROR visibility qualifiers are not permitted here
    pub fn f();
    pub static St: u8;
}

const MAIN: u8 = {
    pub trait Tr {
        fn f();
        const C: u8;
        type T;
    }
    pub struct S {
        pub a: u8
    }
    struct Ts(pub u8);

    pub impl Tr for S {  //~ ERROR visibility qualifiers are not permitted here
        pub fn f() {} //~ ERROR visibility qualifiers are not permitted here
        pub const C: u8 = 0; //~ ERROR visibility qualifiers are not permitted here
        pub type T = u8; //~ ERROR visibility qualifiers are not permitted here
    }
    pub impl S { //~ ERROR visibility qualifiers are not permitted here
        pub fn f() {}
        pub const C: u8 = 0;
        // pub type T = u8;
    }
    pub extern "C" { //~ ERROR visibility qualifiers are not permitted here
        pub fn f();
        pub static St: u8;
    }

    0
};

fn main() {
    pub trait Tr {
        fn f();
        const C: u8;
        type T;
    }
    pub struct S {
        pub a: u8
    }
    struct Ts(pub u8);

    pub impl Tr for S {  //~ ERROR visibility qualifiers are not permitted here
        pub fn f() {} //~ ERROR visibility qualifiers are not permitted here
        pub const C: u8 = 0; //~ ERROR visibility qualifiers are not permitted here
        pub type T = u8; //~ ERROR visibility qualifiers are not permitted here
    }
    pub impl S { //~ ERROR visibility qualifiers are not permitted here
        pub fn f() {}
        pub const C: u8 = 0;
        // pub type T = u8;
    }
    pub extern "C" { //~ ERROR visibility qualifiers are not permitted here
        pub fn f();
        pub static St: u8;
    }
}
