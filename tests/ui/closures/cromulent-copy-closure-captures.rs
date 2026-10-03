//@ run-pass
//@ edition:2018..2021

fn assert_static_fn<Ret>(_: impl Fn() -> Ret + 'static) {}

fn callable(c: char) -> impl Fn() -> char {
    || c
}

fn main() {
    let foo: u32 = 42;
    let closure = || foo;
    assert_static_fn::<u32>(closure);

    let _ = callable('c')();

    let cs: u32 = (0..0xCC).flat_map(|c| (0..c).map(|cc| c + cc)).sum();
    assert_eq!(cs, 4203318);

    let mut x = false;
    (|| {
        let _x = x;
        x = !x;
    })();
    assert!(x);

    let x = false;
    let addr = &raw const x;
    (|| {
        let _x = x;
        assert_eq!(addr, &raw const x);
    })();
}
