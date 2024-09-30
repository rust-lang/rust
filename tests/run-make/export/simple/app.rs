extern dyn crate libr;

fn main() {
    let s = libr::m::S { x: 42 };
    assert_eq!(libr::m::foo1(s), 42);

    assert_eq!(libr::m::foo2(1), 1);
}
