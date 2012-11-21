mod m {
    pub type t = int;
}

macro_rules! foo {
    ($p:path) => ({
        fn f() -> $p { 10 };
        f()
    })
}

fn main() {
    assert foo!(m::t) == 10;
}
