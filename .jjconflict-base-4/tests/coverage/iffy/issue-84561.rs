// This demonstrated Issue #84561: function-like macros produce unintuitive coverage results.

//@ failure-status: 101
#[derive(PartialEq, Eq)]
struct Foo(u32);

#[rustfmt::skip]
fn test3() {
    let is_true = std::env::args().len() == 1;
    let bar = Foo(1);
    assert_eq!(bar, Foo(1));
    let baz = Foo(0);
    assert_ne!(baz, Foo(1));
    println!("{:?}", Foo(1));
    println!("{:?}", bar);
    println!("{:?}", baz);

    assert_eq!(Foo(1), Foo(1));
    assert_ne!(Foo(0), Foo(1));
    assert_eq!(Foo(2), Foo(2));
    let bar = Foo(0);
    assert_ne!(bar, Foo(3));
    assert_ne!(Foo(0), Foo(4));
    assert_eq!(Foo(3), Foo(3), "with a message");
    println!("{:?}", bar);
    println!("{:?}", Foo(1));

    assert_ne!(Foo(0), Foo(5), "{}", if is_true { "true message" } else { "false message" });
    assert_ne!(
        Foo(0)
        ,
        Foo(5)
        ,
        "{}"
        ,
        if
        is_true
        {
            "true message"
        } else {
            "false message"
        }
    );

    let is_true = std::env::args().len() == 1;

    assert_eq!(
        Foo(1),
        Foo(1)
    );
    assert_ne!(
        Foo(0),
        Foo(1)
    );
    assert_eq!(
        Foo(2),
        Foo(2)
    );
    let bar = Foo(1);
    assert_ne!(
        bar,
        Foo(3)
    );
    if is_true {
        assert_ne!(
            Foo(0),
            Foo(4)
        );
    } else {
        assert_eq!(
            Foo(3),
            Foo(3)
        );
    }
    if is_true {
        assert_ne!(
            Foo(0),
            Foo(4),
            "with a message"
        );
    } else {
        assert_eq!(
            Foo(3),
            Foo(3),
            "with a message"
        );
    }
    assert_ne!(
        if is_true {
            Foo(0)
        } else {
            Foo(1)
        },
        Foo(5)
    );
    assert_ne!(
        Foo(5),
        if is_true {
            Foo(0)
        } else {
            Foo(1)
        }
    );
    assert_ne!(
        if is_true {
            assert_eq!(
                Foo(3),
                Foo(3)
            );
            Foo(0)
        } else {
            assert_ne!(
                if is_true {
                    Foo(0)
                } else {
                    Foo(1)
                },
                Foo(5)
            );
            Foo(1)
        },
        Foo(5),
        "with a message"
    );
    assert_eq!(
        Foo(1),
        Foo(3),
        "this assert should fail"
    );
    assert_eq!(
        Foo(3),
        Foo(3),
        "this assert should not be reached"
    );
}

impl std::fmt::Debug for Foo {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "try and succeed")?;
        Ok(())
    }
}

static mut DEBUG_LEVEL_ENABLED: bool = false;

macro_rules! debug {
    ($($arg:tt)+) => (
        if unsafe { DEBUG_LEVEL_ENABLED } {
            println!($($arg)+);
        }
    );
}

fn test1() {
    debug!("debug is enabled");
    debug!("debug is enabled");
    let _ = 0;
    debug!("debug is enabled");
    unsafe {
        DEBUG_LEVEL_ENABLED = true;
    }
    debug!("debug is enabled");
}

macro_rules! call_debug {
    ($($arg:tt)+) => (
        fn call_print(s: &str) {
            print!("{}", s);
        }

        call_print("called from call_debug: ");
        debug!($($arg)+);
    );
}

fn test2() {
    call_debug!("debug is enabled");
}

fn main() {
    test1();
    test2();
    test3();
}
