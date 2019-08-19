// run-pass
#![allow(dead_code)]

pub fn main() {
    let x = Some(3);
    if let Some(y) = x {
        assert_eq!(y, 3);
    } else {
        panic!("if-let panicked");
    }
    let mut worked = false;
    if let Some(_) = x {
        worked = true;
    }
    assert!(worked);
    let clause: usize;
    if let None = Some("test") {
        clause = 1;
    } else if 4_usize > 5 {
        clause = 2;
    } else if let Ok(()) = Err::<(),&'static str>("test") {
        clause = 3;
    } else {
        clause = 4;
    }
    assert_eq!(clause, 4_usize);

    if 3 > 4 {
        panic!("bad math");
    } else if let 1 = 2 {
        panic!("bad pattern match");
    }

    enum Foo {
        One,
        Two(usize),
        Three(String, isize)
    }

    let foo = Foo::Three("three".to_string(), 42);
    if let Foo::One = foo {
        panic!("bad pattern match");
    } else if let Foo::Two(_x) = foo {
        panic!("bad pattern match");
    } else if let Foo::Three(s, _) = foo {
        assert_eq!(s, "three");
    } else {
        panic!("bad else");
    }

    if false {
        panic!("wat");
    } else if let a@Foo::Two(_) = Foo::Two(42_usize) {
        if let Foo::Two(b) = a {
            assert_eq!(b, 42_usize);
        } else {
            panic!("panic in nested if-let");
        }
    }
}
