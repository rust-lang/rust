struct S;

pub mod m {
    fn f() {
        let s = ::m::crate::S; //~ ERROR cannot find module `crate`
        let s1 = ::crate::S; //~ ERROR cannot find module `crate`
        let s2 = crate::S; // no error
    }
}

fn main() {}
