#![feature(optin_builtin_traits)]

pub trait Tr {
    fn f();
    const C: u8;
    type T;
}
pub struct S {
    pub a: u8
}
struct Ts(pub u8);

pub impl Tr for S {  //~ ERROR unnecessary visibility qualifier
    pub fn f() {} //~ ERROR unnecessary visibility qualifier
    pub const C: u8 = 0; //~ ERROR unnecessary visibility qualifier
    pub type T = u8; //~ ERROR unnecessary visibility qualifier
}
pub impl S { //~ ERROR unnecessary visibility qualifier
    pub fn f() {}
    pub const C: u8 = 0;
    // pub type T = u8;
}
pub extern "C" { //~ ERROR unnecessary visibility qualifier
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

    pub impl Tr for S {  //~ ERROR unnecessary visibility qualifier
        pub fn f() {} //~ ERROR unnecessary visibility qualifier
        pub const C: u8 = 0; //~ ERROR unnecessary visibility qualifier
        pub type T = u8; //~ ERROR unnecessary visibility qualifier
    }
    pub impl S { //~ ERROR unnecessary visibility qualifier
        pub fn f() {}
        pub const C: u8 = 0;
        // pub type T = u8;
    }
    pub extern "C" { //~ ERROR unnecessary visibility qualifier
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

    pub impl Tr for S {  //~ ERROR unnecessary visibility qualifier
        pub fn f() {} //~ ERROR unnecessary visibility qualifier
        pub const C: u8 = 0; //~ ERROR unnecessary visibility qualifier
        pub type T = u8; //~ ERROR unnecessary visibility qualifier
    }
    pub impl S { //~ ERROR unnecessary visibility qualifier
        pub fn f() {}
        pub const C: u8 = 0;
        // pub type T = u8;
    }
    pub extern "C" { //~ ERROR unnecessary visibility qualifier
        pub fn f();
        pub static St: u8;
    }
}
