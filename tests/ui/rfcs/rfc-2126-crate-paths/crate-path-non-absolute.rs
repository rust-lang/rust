struct S;

pub mod m {
    fn f() {
        let s = ::m::crate::S; //~ ERROR failed to resolve
        let s1 = ::crate::S; //~ ERROR failed to resolve
        let s2 = crate::S; // no error
    }
}

fn main() {}
