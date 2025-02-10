#![allow(unused_must_use)]

#[allow(dead_code)]
trait Trait {
    fn blah(&self);
}

#[allow(dead_code)]
struct Struct;

impl Trait for Struct {
    cfg_match! {
        cfg(feature = "blah") => {
            fn blah(&self) {
                unimplemented!();
            }
        }
        _ => {
            fn blah(&self) {
                unimplemented!();
            }
        }
    }
}

#[test]
fn assert_eq_trailing_comma() {
    assert_eq!(1, 1,);
}

#[test]
fn assert_escape() {
    assert!(r#"☃\backslash"#.contains("\\"));
}

#[test]
fn assert_ne_trailing_comma() {
    assert_ne!(1, 2,);
}

#[rustfmt::skip]
#[test]
fn matches_leading_pipe() {
    matches!(1, | 1 | 2 | 3);
}

#[test]
fn cfg_match_basic() {
    cfg_match! {
        cfg(target_pointer_width = "64") => { fn f0_() -> bool { true }}
    }

    cfg_match! {
        cfg(unix) => { fn f1_() -> bool { true }}
        cfg(any(target_os = "macos", target_os = "linux")) => { fn f1_() -> bool { false }}
    }

    cfg_match! {
        cfg(target_pointer_width = "32") => { fn f2_() -> bool { false }}
        cfg(target_pointer_width = "64") => { fn f2_() -> bool { true }}
    }

    cfg_match! {
        cfg(target_pointer_width = "16") => { fn f3_() -> i32 { 1 }}
        _ => { fn f3_() -> i32 { 2 }}
    }

    #[cfg(target_pointer_width = "64")]
    assert!(f0_());

    #[cfg(unix)]
    assert!(f1_());

    #[cfg(target_pointer_width = "32")]
    assert!(!f2_());
    #[cfg(target_pointer_width = "64")]
    assert!(f2_());

    #[cfg(not(target_pointer_width = "16"))]
    assert_eq!(f3_(), 2);
}

#[test]
fn cfg_match_debug_assertions() {
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

#[cfg(target_pointer_width = "64")]
#[test]
fn cfg_match_no_duplication_on_64() {
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
    foo();
}

#[test]
fn cfg_match_options() {
    cfg_match! {
        cfg(test) => {
            use core::option::Option as Option2;
            fn works1() -> Option2<u32> { Some(1) }
        }
        _ => { fn works1() -> Option<u32> { None } }
    }

    cfg_match! {
        cfg(feature = "foo") => { fn works2() -> bool { false } }
        cfg(test) => { fn works2() -> bool { true } }
        _ => { fn works2() -> bool { false } }
    }

    cfg_match! {
        cfg(feature = "foo") => { fn works3() -> bool { false } }
        _ => { fn works3() -> bool { true } }
    }

    cfg_match! {
        cfg(test) => {
            use core::option::Option as Option3;
            fn works4() -> Option3<u32> { Some(1) }
        }
    }

    cfg_match! {
        cfg(feature = "foo") => { fn works5() -> bool { false } }
        cfg(test) => { fn works5() -> bool { true } }
    }

    assert!(works1().is_some());
    assert!(works2());
    assert!(works3());
    assert!(works4().is_some());
    assert!(works5());
}

#[test]
fn cfg_match_two_functions() {
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

fn _accepts_expressions() -> i32 {
    cfg_match! {
        cfg(unix) => { 1 }
        _ => { 2 }
    }
}

fn _allows_stmt_expr_attributes() {
    let one = 1;
    let two = 2;
    cfg_match! {
        cfg(unix) => { one * two; }
        _ => { one + two; }
    }
}
