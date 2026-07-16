//@aux-build:proc_macros.rs

#![feature(never_type)]
#![warn(clippy::redundant_else)]
#![expect(clippy::unnecessary_operation)]

extern crate proc_macros;
use proc_macros::{external, inline_macros, with_span};

use core::hint::black_box;
use core::ops::{Add, Deref, Not};

#[inline_macros]
fn main() {
    // then syntactic diverge final expr.
    {
        if black_box(false) {
            return;
        } else {
            //~^ redundant_else
            black_box(0)
        };

        loop {
            if black_box(false) {
                break;
            } else {
                //~^ redundant_else
                black_box(0)
            };
        }

        loop {
            if black_box(false) {
                continue;
            } else {
                //~^ redundant_else
                black_box(0)
            };
        }
    }

    // then syntactic diverge final stmt.
    {
        if black_box(false) {
            return;
        } else {
            //~^ redundant_else
            black_box(0)
        };

        loop {
            if black_box(false) {
                break;
            } else {
                //~^ redundant_else
                black_box(0)
            };
            if black_box(false) {
                continue;
            } else {
                //~^ redundant_else
                black_box(0)
            };
        }
    }

    // then panic
    {
        if black_box(false) {
            panic!();
        } else {
            //~^ redundant_else
            black_box(())
        };

        if black_box(false) {
            panic!("msg");
        } else {
            //~^ redundant_else
            black_box(0)
        };

        if black_box(false) {
            panic!("{}", 0);
        } else {
            //~^ redundant_else
            black_box(0i32)
        };
    };

    // no semi on if
    {
        if black_box(false) {
            return;
        } else {
            //~^ redundant_else
            black_box(0);
        }

        if black_box(false) {
            return;
        } else {
            //~^ redundant_else
            let _ = 0;
            black_box(())
        }

        if black_box(false) {
            return;
        } else {
            //~^ redundant_else
            { black_box(()) }
        }

        if black_box(false) {
            return;
        } else {
            //~^ redundant_else
            if black_box(true) {
                black_box(0);
            }
        }

        if black_box(false) {
            return;
        } else {
            //~^ redundant_else
            if black_box(true) {
                // empty
            } else {
                black_box(())
            }
        }

        if black_box(false) {
            return;
        } else {
            //~^ redundant_else
            match black_box(0) {
                0 => {},
                1 => {
                    let _ = 0;
                },
                _ => black_box(()),
            }
        }

        if black_box(false) {
            return;
        } else {
            //~^ redundant_else
            loop {
                if black_box(true) {
                    break black_box(());
                }
            }
        }

        if black_box(false) {
            return;
        } else {
            //~^ redundant_else
            while black_box(true) {
                if black_box(true) {
                    break;
                }
            }
        }

        if black_box(false) {
            return;
        } else {
            //~^ redundant_else
            for _ in black_box(0..0) {
                if black_box(true) {
                    break;
                }
            }
        }

        if black_box(false) {
            return;
        } else {
            //~^ redundant_else
            black_box(1)
        }
    };

    // then nested block diverge
    {
        if black_box(false) {
            {
                let _ = black_box(0);
                panic!()
            }
        } else {
            //~^ redundant_else
            black_box(0)
        };

        if black_box(false) {
            {
                let _ = black_box(0);
                return;
            }
        } else {
            //~^ redundant_else
            black_box(0)
        };
    }

    // then nested branch diverge
    {
        loop {
            if black_box(false) {
                match black_box(0) {
                    0 => return,
                    1 => panic!(),
                    2 => break,
                    _ => continue,
                }
            } else {
                //~^ redundant_else
                black_box(0)
            };
        }

        if black_box(false) {
            if black_box(true) {
                panic!();
            } else {
                //~^ redundant_else
                return;
            }
        } else {
            //~^ redundant_else
            black_box(0)
        };
    }

    // if chain diverge
    {
        if black_box(false) {
            panic!()
        } else if black_box(true) {
            return;
        } else if black_box(true) {
            let x = 0;
            black_box(x + 2 * 55);
            panic!();
        } else {
            //~^ redundant_else
            black_box(0)
        };
    }

    // then misc fn diverge
    {
        struct S;
        impl Deref for S {
            type Target = !;
            fn deref(&self) -> &! {
                panic!()
            }
        };
        impl Not for S {
            type Output = !;
            fn not(self) -> ! {
                panic!()
            }
        }
        impl Add for S {
            type Output = !;
            fn add(self, other: Self) -> ! {
                panic!()
            }
        }

        if black_box(true) {
            *S
        } else if black_box(true) {
            !S
        } else if black_box(true) {
            S + S
        } else if black_box(true) {
            S.not()
        } else if black_box(true) {
            S::add(S, S)
        } else {
            //~^ redundant_else
            black_box(0)
        };

        if black_box(true) {
            *S;
        } else if black_box(true) {
            !S;
        } else if black_box(true) {
            S + S;
        } else if black_box(true) {
            S.not();
        } else if black_box(true) {
            S::add(S, S);
        } else {
            //~^ redundant_else
            black_box(0)
        };
    }

    // then no diverge
    {
        if black_box(true) {
            black_box(1)
        } else {
            black_box(0)
        };

        if black_box(true) {
            if black_box(true) { black_box(0) } else { return }
        } else {
            black_box(0)
        };

        if black_box(true) {
            if black_box(true) { return } else { black_box(0) }
            //~^ redundant_else
        } else {
            black_box(0)
        };

        loop {
            if black_box(true) {
                match black_box(1) {
                    0 => panic!(),
                    1 => return,
                    2 => break,
                    _ => {
                        if black_box(true) {
                            break;
                        } else if black_box(true) {
                            panic!()
                        } else if black_box(true) {
                            black_box(0)
                        } else {
                            return;
                        }
                    },
                }
            } else {
                black_box(0)
            };
        }
    }

    // nested in various weird positions
    {
        black_box(if black_box(true) {
            return;
        } else {
            black_box(0)
        });

        let _ = if black_box(true) {
            return;
        } else {
            black_box(0)
        };

        let _ = black_box([0, 1, 2].as_slice())[if black_box(true) {
            return;
        } else {
            black_box(0)
        }];

        (if black_box(true) {
            return;
        } else {
            black_box(0i32)
        })
        .ilog2();

        return if black_box(true) {
            return;
        } else {
            black_box(())
        };

        1 + if black_box(true) {
            return;
        } else {
            black_box(0)
        };
    }

    // external macros
    {
        external! {{
            if black_box(true) {
                return
            } else {
                black_box(0)
            };
        }}
        with_span! {
            span
            {
                if black_box(true) {
                    return
                } else {
                    black_box(0)
                };
            }
        }
    }

    // internal macros
    {
        fn diverge() -> ! {
            panic!()
        }

        inline! {{
            if black_box(true) {
                return
            } else {
                //~^ redundant_else
                black_box(0);
            }

            if black_box(true) {
                return;
            } else {
                //~^ redundant_else
                black_box(0);
            }

            if black_box(true) {
                if black_box(true) {
                    return;
                } else {
                    //~^ redundant_else
                    let _ = ();
                    return
                }
            } else {
                //~^ redundant_else
                black_box(0);
            }

            if black_box(true) {
                #[expect(clippy::redundant_else)]
                if black_box(true) {
                    return;
                } else {
                    let _ = ();
                    return;
                };
            } else {
                //~^ redundant_else
                black_box(0);
            }

            if black_box(true) {
                diverge();
            } else {
                black_box(0);
            }

            // Needs a semicolon in the suggestion due to the macro call
            if black_box(true) {
                return
            } else {
                //~^ redundant_else
                inline!({ black_box(()); })
            }

            // Needs a semicolon in the suggestion due to the context switch
            if black_box(true) {
                return
            } else {
                //~^ redundant_else
                $({ black_box(()) })
            }

            let _ = 0;
        }};
    }
}
