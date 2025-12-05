#![warn(clippy::match_ref_pats)]
#![allow(dead_code, unused_variables)]
#![allow(
    clippy::enum_variant_names,
    clippy::equatable_if_let,
    clippy::uninlined_format_args,
    clippy::empty_loop,
    clippy::diverging_sub_expression
)]

fn ref_pats() {
    {
        let v = &Some(0);
        match v {
            //~^ match_ref_pats
            &Some(v) => println!("{:?}", v),
            &None => println!("none"),
        }
        match v {
            // This doesn't trigger; we have a different pattern.
            &Some(v) => println!("some"),
            other => println!("other"),
        }
    }
    let tup = &(1, 2);
    match tup {
        &(v, 1) => println!("{}", v),
        _ => println!("none"),
    }
    // Special case: using `&` both in expr and pats.
    let w = Some(0);
    match &w {
        //~^ match_ref_pats
        &Some(v) => println!("{:?}", v),
        &None => println!("none"),
    }
    // False positive: only wildcard pattern.
    let w = Some(0);
    #[allow(clippy::match_single_binding)]
    match w {
        _ => println!("none"),
    }

    let a = &Some(0);
    if let &None = a {
        //~^ redundant_pattern_matching
        println!("none");
    }

    let b = Some(0);
    if let &None = &b {
        //~^ redundant_pattern_matching
        println!("none");
    }
}

mod ice_3719 {
    macro_rules! foo_variant(
        ($idx:expr) => (Foo::get($idx).unwrap())
    );

    enum Foo {
        A,
        B,
    }

    impl Foo {
        fn get(idx: u8) -> Option<&'static Self> {
            match idx {
                0 => Some(&Foo::A),
                1 => Some(&Foo::B),
                _ => None,
            }
        }
    }

    fn ice_3719() {
        // ICE #3719
        match foo_variant!(0) {
            &Foo::A => println!("A"),
            _ => println!("Wild"),
        }
    }
}

mod issue_7740 {
    macro_rules! foobar_variant(
        ($idx:expr) => (FooBar::get($idx).unwrap())
    );

    enum FooBar {
        Foo,
        Bar,
        FooBar,
        BarFoo,
    }

    impl FooBar {
        fn get(idx: u8) -> Option<&'static Self> {
            match idx {
                0 => Some(&FooBar::Foo),
                1 => Some(&FooBar::Bar),
                2 => Some(&FooBar::FooBar),
                3 => Some(&FooBar::BarFoo),
                _ => None,
            }
        }
    }

    fn issue_7740() {
        // Issue #7740
        match foobar_variant!(0) {
            //~^ match_ref_pats
            &FooBar::Foo => println!("Foo"),
            &FooBar::Bar => println!("Bar"),
            &FooBar::FooBar => println!("FooBar"),
            _ => println!("Wild"),
        }

        // This shouldn't trigger
        if let &FooBar::BarFoo = foobar_variant!(3) {
            println!("BarFoo");
        } else {
            println!("Wild");
        }
    }
}

mod issue15378 {
    fn never_in_match() {
        match unimplemented!() {
            &_ => {},
            &&&42 => {
                todo!()
            },
            _ => {},
        }

        match panic!() {
            &_ => {},
            &&&42 => {
                todo!()
            },
            _ => {},
        }

        match loop {} {
            &_ => {},
            &&&42 => {
                todo!()
            },
            _ => {},
        }
    }
}

fn main() {}
