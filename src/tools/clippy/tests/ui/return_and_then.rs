#![warn(clippy::return_and_then)]

fn main() {
    fn test_opt_block(opt: Option<i32>) -> Option<i32> {
        opt.and_then(|n| {
            //~^ return_and_then
            let mut ret = n + 1;
            ret += n;
            if n > 1 { Some(ret) } else { None }
        })
    }

    fn test_opt_func(opt: Option<i32>) -> Option<i32> {
        opt.and_then(|n| test_opt_block(Some(n)))
        //~^ return_and_then
    }

    fn test_call_chain() -> Option<i32> {
        gen_option(1).and_then(|n| test_opt_block(Some(n)))
        //~^ return_and_then
    }

    fn test_res_block(opt: Result<i32, i32>) -> Result<i32, i32> {
        opt.and_then(|n| if n > 1 { Ok(n + 1) } else { Err(n) })
        //~^ return_and_then
    }

    fn test_res_func(opt: Result<i32, i32>) -> Result<i32, i32> {
        opt.and_then(|n| test_res_block(Ok(n)))
        //~^ return_and_then
    }

    fn test_ref_only() -> Option<i32> {
        // ref: empty string
        Some("").and_then(|x| if x.len() > 2 { Some(3) } else { None })
        //~^ return_and_then
    }

    fn test_tmp_only() -> Option<i32> {
        // unused temporary: vec![1, 2, 4]
        Some(match (vec![1, 2, 3], vec![1, 2, 4]) {
            //~^ return_and_then
            (a, _) if a.len() > 1 => a,
            (_, b) => b,
        })
        .and_then(|x| if x.len() > 2 { Some(3) } else { None })
    }

    // should not lint
    fn test_tmp_ref() -> Option<String> {
        String::from("<BOOM>")
            .strip_prefix("<")
            .and_then(|s| s.strip_suffix(">").map(String::from))
    }

    // should not lint
    fn test_unconsumed_tmp() -> Option<i32> {
        [1, 2, 3]
            .iter()
            .map(|x| x + 1)
            .collect::<Vec<_>>() // temporary Vec created here
            .as_slice() // creates temporary slice
            .first() // creates temporary reference
            .and_then(|x| test_opt_block(Some(*x)))
    }

    fn in_closure() -> bool {
        let _ = || {
            Some("").and_then(|x| if x.len() > 2 { Some(3) } else { None })
            //~^ return_and_then
        };
        true
    }

    fn with_return(shortcut: bool) -> Option<i32> {
        if shortcut {
            return Some("").and_then(|x| if x.len() > 2 { Some(3) } else { None });
            //~^ return_and_then
        };
        None
    }

    fn with_return_multiline(shortcut: bool) -> Option<i32> {
        if shortcut {
            return Some("").and_then(|mut x| {
                let x = format!("{x}.");
                if x.len() > 2 { Some(3) } else { None }
            });
            //~^^^^ return_and_then
        };
        None
    }

    #[expect(clippy::diverging_sub_expression)]
    fn with_return_in_expression() -> Option<i32> {
        _ = (
            return Some("").and_then(|x| if x.len() > 2 { Some(3) } else { None }),
            //~^ return_and_then
            10,
        );
    }

    fn inside_if(a: bool, i: Option<u32>) -> Option<u32> {
        if a {
            i.and_then(|i| if i > 3 { Some(i) } else { None })
            //~^ return_and_then
        } else {
            Some(42)
        }
    }

    fn inside_match(a: u32, i: Option<u32>) -> Option<u32> {
        match a {
            1 | 2 => i.and_then(|i| if i > 3 { Some(i) } else { None }),
            //~^ return_and_then
            3 | 4 => Some(42),
            _ => None,
        }
    }

    fn inside_match_and_block_and_if(a: u32, i: Option<u32>) -> Option<u32> {
        match a {
            1 | 2 => {
                let a = a * 3;
                if a.is_multiple_of(2) {
                    i.and_then(|i| if i > 3 { Some(i) } else { None })
                    //~^ return_and_then
                } else {
                    Some(10)
                }
            },
            3 | 4 => Some(42),
            _ => None,
        }
    }

    #[expect(clippy::never_loop)]
    fn with_break(i: Option<u32>) -> Option<u32> {
        match i {
            Some(1) => loop {
                break i.and_then(|i| if i > 3 { Some(i) } else { None });
                //~^ return_and_then
            },
            Some(2) => 'foo: loop {
                loop {
                    break 'foo i.and_then(|i| if i > 3 { Some(i) } else { None });
                    //~^ return_and_then
                }
            },
            Some(3) => 'bar: {
                break 'bar i.and_then(|i| if i > 3 { Some(i) } else { None });
                //~^ return_and_then
            },
            Some(4) => 'baz: loop {
                _ = loop {
                    break i.and_then(|i| if i > 3 { Some(i) } else { None });
                };
            },
            _ => None,
        }
    }
}

fn gen_option(n: i32) -> Option<i32> {
    Some(n)
}

mod issue14781 {
    fn foo(_: &str, _: (u32, u32)) -> Result<(u32, u32), ()> {
        Ok((1, 1))
    }

    fn bug(_: Option<&str>) -> Result<(), ()> {
        let year: Option<&str> = None;
        let month: Option<&str> = None;
        let day: Option<&str> = None;

        let _day = if let (Some(year), Some(month)) = (year, month) {
            day.and_then(|day| foo(day, (1, 31)).ok())
        } else {
            None
        };

        Ok(())
    }
}

mod issue15111 {
    #[derive(Debug)]
    struct EvenOdd {
        even: Option<u32>,
        odd: Option<u32>,
    }

    impl EvenOdd {
        fn new(i: Option<u32>) -> Self {
            Self {
                even: i.and_then(|i| if i.is_multiple_of(2) { Some(i) } else { None }),
                odd: i.and_then(|i| if i.is_multiple_of(2) { None } else { Some(i) }),
            }
        }
    }

    fn with_if_let(i: Option<u32>) -> u32 {
        if let Some(x) = i.and_then(|i| if i.is_multiple_of(2) { Some(i) } else { None }) {
            x
        } else {
            std::hint::black_box(0)
        }
    }

    fn main() {
        let _ = EvenOdd::new(Some(2));
    }
}

mod issue14927 {
    use std::path::Path;
    struct A {
        pub func: fn(check: bool, a: &Path, b: Option<&Path>),
    }
    const MY_A: A = A {
        func: |check, a, b| {
            if check {
                let _ = ();
            } else if let Some(parent) = b.and_then(|p| p.parent()) {
                let _ = ();
            }
        },
    };
}
