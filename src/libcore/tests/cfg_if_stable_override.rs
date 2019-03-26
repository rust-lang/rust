//! Test that using cfg_if from the cfg_if crate does not break anything.

#[macro_use(cfg_if)]
extern crate cfg_if;

cfg_if! {
    if #[cfg(test)] {
        use core::option::Option as Option2;
        fn works1() -> Option2<u32> { Some(1) }
    } else {
        fn works1() -> Option<u32> { None }
    }
}

cfg_if! {
    if #[cfg(foo)] {
        fn works2() -> bool { false }
    } else if #[cfg(test)] {
        fn works2() -> bool { true }
    } else {
        fn works2() -> bool { false }
    }
}

cfg_if! {
    if #[cfg(foo)] {
        fn works3() -> bool { false }
    } else {
        fn works3() -> bool { true }
    }
}

cfg_if! {
    if #[cfg(test)] {
        use core::option::Option as Option3;
        fn works4() -> Option3<u32> { Some(1) }
    }
}

cfg_if! {
    if #[cfg(foo)] {
        fn works5() -> bool { false }
    } else if #[cfg(test)] {
        fn works5() -> bool { true }
    }
}

#[test]
fn it_works() {
    assert!(works1().is_some());
    assert!(works2());
    assert!(works3());
    assert!(works4().is_some());
    assert!(works5());
}
