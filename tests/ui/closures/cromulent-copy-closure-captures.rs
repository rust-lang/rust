//@ run-pass
//@ revisions: edition2018 edition2021
//@ [edition2018] edition:2018
//@ [edition2021] edition:2021

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

    let mut x = 2i32;
    let ref_mut = &mut x;
    (|| {
        *ref_mut = 42;
    })();
    assert_eq!(*ref_mut, 42);

    let y: i32 = 5;
    let closure = |x: i32| x + y;
    assert_eq!(size_of_val(&closure), 4);

    {
        let val = 21;
        let raw_ptr = &raw const val;
        let closure = || assert_eq!(unsafe { *raw_ptr }, 21);
        assert_static_fn(closure);
    }

    {
        let mut val = (21,);
        let raw_mut_ptr = &raw mut val;
        let closure = || unsafe {
            (*raw_mut_ptr).0 = 42;
        };
        assert_static_fn(closure);
    }

    {
        let val = 21;
        let closure = || { val } == 21;
        assert_static_fn(closure);
    }
}
