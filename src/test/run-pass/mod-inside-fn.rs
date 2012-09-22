fn f() -> int {
    mod m {
        #[legacy_exports];
        fn g() -> int { 720 }
    }

    m::g()
}

fn main() {
    assert f() == 720;
}