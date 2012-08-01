fn f() -> int {
    mod m {
        fn g() -> int { 720 }
    }

    m::g()
}

fn main() {
    assert f() == 720;
}