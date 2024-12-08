#![warn(clippy::manual_inspect)]
#![allow(clippy::no_effect, clippy::op_ref)]

fn main() {
    let _ = Some(0).map(|x| {
        println!("{}", x);
        x
    });

    let _ = Some(0).map(|x| {
        println!("{x}");
        x
    });

    let _ = Some(0).map(|x| {
        println!("{}", x * 5 + 1);
        x
    });

    let _ = Some(0).map(|x| {
        if x == 0 {
            panic!();
        }
        x
    });

    let _ = Some(0).map(|x| {
        if &x == &0 {
            let _y = x;
            panic!();
        }
        x
    });

    let _ = Some(0).map(|x| {
        let y = x + 1;
        if y > 5 {
            return y;
        }
        x
    });

    {
        #[derive(PartialEq)]
        struct Foo(i32);

        let _ = Some(Foo(0)).map(|x| {
            if x == Foo(0) {
                panic!();
            }
            x
        });

        let _ = Some(Foo(0)).map(|x| {
            if &x == &Foo(0) {
                let _y = x;
                panic!();
            }
            x
        });
    }

    {
        macro_rules! maybe_ret {
            ($e:expr) => {
                if $e == 0 {
                    return $e;
                }
            };
        }

        let _ = Some(0).map(|x| {
            maybe_ret!(x);
            x
        });
    }

    let _ = Some((String::new(), 0u32)).map(|x| {
        if x.1 == 0 {
            let _x = x.1;
            panic!();
        }
        x
    });

    let _ = Some((String::new(), 0u32)).map(|x| {
        if x.1 == 0 {
            let _x = x.0;
            panic!();
        }
        x
    });

    let _ = Some(String::new()).map(|x| {
        if x.is_empty() {
            let _ = || {
                let _x = x;
            };
            panic!();
        }
        x
    });

    let _ = Some(String::new()).map(|x| {
        if x.is_empty() {
            let _ = || {
                let _x = &x;
            };
            return x;
        }
        println!("test");
        x
    });

    let _ = Some(0).map(|x| {
        if x == 0 {
            let _ = || {
                let _x = x;
            };
            panic!();
        }
        x
    });

    {
        use core::cell::Cell;
        #[derive(Debug)]
        struct Cell2(core::cell::Cell<u32>);

        let _ = Some(Cell2(Cell::new(0u32))).map(|x| {
            x.0.set(1);
            x
        });

        let _ = Some(Cell2(Cell::new(0u32))).map(|x| {
            let y = &x;
            if x.0.get() == 0 {
                y.0.set(1)
            } else {
                println!("{x:?}");
            }
            x
        });
    }

    let _: Result<_, ()> = Ok(0).map(|x| {
        println!("{}", x);
        x
    });

    let _: Result<(), _> = Err(0).map_err(|x| {
        println!("{}", x);
        x
    });

    let _ = [0]
        .into_iter()
        .map(|x| {
            println!("{}", x);
            x
        })
        .count();

    {
        struct S<T>(T);
        impl<T> S<T> {
            fn map<U>(self, f: impl FnOnce(T) -> U) -> S<U> {
                S(f(self.0))
            }

            fn map_err<U>(self, f: impl FnOnce(T) -> U) -> S<U> {
                S(f(self.0))
            }
        }

        let _ = S(0).map(|x| {
            println!("{}", x);
            x
        });

        let _ = S(0).map_err(|x| {
            println!("{}", x);
            x
        });
    }
}
