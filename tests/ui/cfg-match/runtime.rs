// run-pass

#![feature(cfg_match)]

fn basic() {
    cfg_match! {
        cfg(unix) => { fn f1() -> bool { true }}
        cfg(any(target_os = "macos", target_os = "linux")) => { fn f1() -> bool { false }}
    }

    cfg_match! {
        cfg(target_pointer_width = "32") => { fn f2() -> bool { false }}
        cfg(target_pointer_width = "64") => { fn f2() -> bool { true }}
    }

    cfg_match! {
        cfg(target_pointer_width = "16") => { fn f3() -> i32 { 1 }}
        _ => { fn f3() -> i32 { 2 }}
    }

    cfg_match! {
        cfg(test) => {
            use core::option::Option as Option2;
            fn works1() -> Option2<u32> { Some(1) }
        }
        _ => { fn works1() -> Option<u32> { None } }
    }

    cfg_match! {
        cfg(feature = "foo") => { fn works2() -> bool { false } }
        cfg(test) => { fn works2() -> bool { false } }
        _ => { fn works2() -> bool { true } }
    }

    cfg_match! {
        cfg(feature = "foo") => { fn works3() -> bool { false } }
        _ => { fn works3() -> bool { true } }
    }

    #[cfg(unix)]
    assert!(f1());

    #[cfg(target_pointer_width = "32")]
    assert!(!f2());
    #[cfg(target_pointer_width = "64")]
    assert!(f2());

    #[cfg(not(target_pointer_width = "16"))]
    assert_eq!(f3(), 2);

    assert!(works1().is_none());

    assert!(works2());

    assert!(works3());
}

fn debug_assertions() {
    cfg_match! {
        cfg(debug_assertions) => {
            assert!(cfg!(debug_assertions));
            assert_eq!(4, 2+2);
        }
        _ => {
            assert!(cfg!(not(debug_assertions)));
            assert_eq!(10, 5+5);
        }
    }
}

fn no_bracket() {
    cfg_match! {
        cfg(unix) => fn f0() -> bool { true },
        _ => fn f0() -> bool { true },
    }
    assert!(f0())
}

fn no_duplication_on_64() {
    #[cfg(target_pointer_width = "64")]
    cfg_match! {
        cfg(windows) => {
            fn foo() {}
        }
        cfg(unix) => {
            fn foo() {}
        }
        cfg(target_pointer_width = "64") => {
            fn foo() {}
        }
    }
    #[cfg(target_pointer_width = "64")]
    foo();
}

fn two_functions() {
    cfg_match! {
        cfg(target_pointer_width = "64") => {
            fn foo1() {}
            fn bar1() {}
        }
        _ => {
            fn foo2() {}
            fn bar2() {}
        }
    }

    #[cfg(target_pointer_width = "64")]
    {
        foo1();
        bar1();
    }
    #[cfg(not(target_pointer_width = "64"))]
    {
        foo2();
        bar2();
    }
}

pub fn main() {
    basic();
    debug_assertions();
    no_bracket();
    no_duplication_on_64();
    two_functions();
}
