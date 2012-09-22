use m::f;
use m::g;

mod m {
    #[legacy_exports];
    export f, g;

    fn f() { }
    fn g() { }
}

fn main() { f(); g(); m::f(); m::g(); }
