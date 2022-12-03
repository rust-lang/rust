// run-rustfix

#![warn(clippy::match_like_matches_macro)]
#![allow(
    unreachable_patterns,
    dead_code,
    clippy::equatable_if_let,
    clippy::needless_borrowed_reference
)]

fn main() {
    let x = Some(5);

    // Lint
    let _y = match x {
        Some(0) => true,
        _ => false,
    };

    // Lint
    let _w = match x {
        Some(_) => true,
        _ => false,
    };

    // Turn into is_none
    let _z = match x {
        Some(_) => false,
        None => true,
    };

    // Lint
    let _zz = match x {
        Some(r) if r == 0 => false,
        _ => true,
    };

    // Lint
    let _zzz = if let Some(5) = x { true } else { false };

    // No lint
    let _a = match x {
        Some(_) => false,
        _ => false,
    };

    // No lint
    let _ab = match x {
        Some(0) => false,
        _ => true,
        None => false,
    };

    enum E {
        A(u32),
        B(i32),
        C,
        D,
    }
    let x = E::A(2);
    {
        // lint
        let _ans = match x {
            E::A(_) => true,
            E::B(_) => true,
            _ => false,
        };
    }
    {
        // lint
        // skip rustfmt to prevent removing block for first pattern
        #[rustfmt::skip]
        let _ans = match x {
            E::A(_) => {
                true
            }
            E::B(_) => true,
            _ => false,
        };
    }
    {
        // lint
        let _ans = match x {
            E::B(_) => false,
            E::C => false,
            _ => true,
        };
    }
    {
        // no lint
        let _ans = match x {
            E::A(_) => false,
            E::B(_) => false,
            E::C => true,
            _ => true,
        };
    }
    {
        // no lint
        let _ans = match x {
            E::A(_) => true,
            E::B(_) => false,
            E::C => false,
            _ => true,
        };
    }
    {
        // no lint
        let _ans = match x {
            E::A(a) if a < 10 => false,
            E::B(a) if a < 10 => false,
            _ => true,
        };
    }
    {
        // no lint
        let _ans = match x {
            E::A(_) => false,
            E::B(a) if a < 10 => false,
            _ => true,
        };
    }
    {
        // no lint
        let _ans = match x {
            E::A(a) => a == 10,
            E::B(_) => false,
            _ => true,
        };
    }
    {
        // no lint
        let _ans = match x {
            E::A(_) => false,
            E::B(_) => true,
            _ => false,
        };
    }

    {
        // should print "z" in suggestion (#6503)
        let z = &Some(3);
        let _z = match &z {
            Some(3) => true,
            _ => false,
        };
    }

    {
        // this could also print "z" in suggestion..?
        let z = Some(3);
        let _z = match &z {
            Some(3) => true,
            _ => false,
        };
    }

    {
        enum AnEnum {
            X,
            Y,
        }

        fn foo(_x: AnEnum) {}

        fn main() {
            let z = AnEnum::X;
            // we can't remove the reference here!
            let _ = match &z {
                AnEnum::X => true,
                _ => false,
            };
            foo(z);
        }
    }

    {
        struct S(i32);

        fn fun(_val: Option<S>) {}
        let val = Some(S(42));
        // we need the reference here because later val is consumed by fun()
        let _res = match &val {
            &Some(ref _a) => true,
            _ => false,
        };
        fun(val);
    }

    {
        struct S(i32);

        fn fun(_val: Option<S>) {}
        let val = Some(S(42));
        let _res = match &val {
            &Some(ref _a) => true,
            _ => false,
        };
        fun(val);
    }

    {
        enum E {
            A,
            B,
            C,
        }

        let _ = match E::A {
            E::B => true,
            #[cfg(feature = "foo")]
            E::A => true,
            _ => false,
        };
    }

    let x = ' ';
    // ignore if match block contains comment
    let _line_comments = match x {
        // numbers are bad!
        '1' | '2' | '3' => true,
        // spaces are very important to be true.
        ' ' => true,
        // as are dots
        '.' => true,
        _ => false,
    };

    let _block_comments = match x {
        /* numbers are bad!
         */
        '1' | '2' | '3' => true,
        /* spaces are very important to be true.
         */
        ' ' => true,
        /* as are dots
         */
        '.' => true,
        _ => false,
    };
}

#[clippy::msrv = "1.41"]
fn msrv_1_41() {
    let _y = match Some(5) {
        Some(0) => true,
        _ => false,
    };
}

#[clippy::msrv = "1.42"]
fn msrv_1_42() {
    let _y = match Some(5) {
        Some(0) => true,
        _ => false,
    };
}
