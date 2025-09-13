struct S;

pub mod m {
    fn f() {
        let s = ::m::crate::S; //~ ERROR: `crate` in paths can only be used in start position
        let s1 = ::crate::S; //~ ERROR: global paths cannot start with `crate`
        let s2 = crate::S; // no error
    }
}

fn main() {}
