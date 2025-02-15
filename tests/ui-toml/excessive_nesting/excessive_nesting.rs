//@aux-build:../../ui/auxiliary/proc_macros.rs
#![rustfmt::skip]
#![feature(custom_inner_attributes)]
#![warn(clippy::excessive_nesting)]
#![allow(
    unused,
    clippy::let_and_return,
    clippy::redundant_closure_call,
    clippy::no_effect,
    clippy::unnecessary_operation,
    clippy::never_loop,
    clippy::needless_if,
    clippy::collapsible_if,
    clippy::blocks_in_conditions,
    clippy::single_match,
)]

#[macro_use]
extern crate proc_macros;

static X: u32 = {
    let x = {
        let y = {
            let z = {
                let w = { 3 };
                //~^ excessive_nesting
                w
            };
            z
        };
        y
    };
    x
};

macro_rules! xx {
    () => {{
        {
            {
                {
                    {
                        {
                            {
                                {
                                    {
                                        {
                                            {
                                                println!("ehe"); // should not lint
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }};
}

struct A;

impl A {
    pub fn a(&self, v: u32) {
        struct B;

        impl B {
            pub fn b() {
                struct C;

                impl C {
                //~^ excessive_nesting
                    pub fn c() {}
                }
            }
        }
    }
}

struct D { d: u32 }

trait Lol {
    fn lmao() {
        fn bb() {
            fn cc() {
                let x = { 1 }; // not a warning, but cc is
                //~^ excessive_nesting
            }

            let x = { 1 }; // warning
        }
    }
}

#[allow(clippy::excessive_nesting)]
fn l() {{{{{{{{{}}}}}}}}}

use a::{b::{c::{d::{e::{f::{}}}}}}; // should not lint

pub mod a {
    pub mod b {
        pub mod c {
            pub mod d {
                pub mod e {
                //~^ excessive_nesting
                    pub mod f {}
                } // not here
            } // only warning should be here
        }
    }
}

fn a_but_not(v: u32) {}

fn main() {
    let a = A;

    a_but_not({{{{{{{{0}}}}}}}});
    //~^ excessive_nesting
    a.a({{{{{{{{{0}}}}}}}}});
    //~^ excessive_nesting
    (0, {{{{{{{1}}}}}}});
    //~^ excessive_nesting

    if true {
        if true {
            if true {
                if true {
                //~^ excessive_nesting
                    if true {

                    }
                }
            }
        }
    }

    let y = (|| {
        let x = (|| {
            let y = (|| {
                let z = (|| {
                //~^ excessive_nesting
                    let w = { 3 };
                    w
                })();
                z
            })();
            y
        })();
        x
    })();

    external! { {{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}} }; // ensure this isn't linted in external macros
    with_span! { span {{{{{{{{{{{{}}}}}}}}}}}} }; // don't lint for proc macros
    xx!(); // ensure this is never linted
    let boo = true;
    !{boo as u32 + !{boo as u32 + !{boo as u32}}};

    // this is a mess, but that's intentional
    let mut y = 1;
    y += {{{{{5}}}}};
    //~^ excessive_nesting
    let z = y + {{{{{{{{{5}}}}}}}}};
    //~^ excessive_nesting
    [0, {{{{{{{{{{0}}}}}}}}}}];
    //~^ excessive_nesting
    let mut xx = [0; {{{{{{{{100}}}}}}}}];
    //~^ excessive_nesting
    xx[{{{{{{{{{{{{{{{{{{{{{{{{3}}}}}}}}}}}}}}}}}}}}}}}}];
    //~^ excessive_nesting
    &mut {{{{{{{{{{y}}}}}}}}}};
    //~^ excessive_nesting

    for i in {{{{xx}}}} {{{{{{{{}}}}}}}}
    //~^ excessive_nesting
    //~| excessive_nesting

    while let Some(i) = {{{{{{Some(1)}}}}}} {{{{{{{}}}}}}}
    //~^ excessive_nesting
    //~| excessive_nesting

    while {{{{{{{{true}}}}}}}} {{{{{{{{{}}}}}}}}}
    //~^ excessive_nesting
    //~| excessive_nesting

    let d = D { d: {{{{{{{{{{{{{{{{{{{{{{{3}}}}}}}}}}}}}}}}}}}}}}} };
    //~^ excessive_nesting

    {{{{1;}}}}..{{{{{{3}}}}}};
    //~^ excessive_nesting
    //~| excessive_nesting
    {{{{1;}}}}..={{{{{{{{{{{{{{{{{{{{{{{{{{6}}}}}}}}}}}}}}}}}}}}}}}}}};
    //~^ excessive_nesting
    //~| excessive_nesting
    ..{{{{{{{5}}}}}}};
    //~^ excessive_nesting
    ..={{{{{3}}}}};
    //~^ excessive_nesting
    {{{{{1;}}}}}..;
    //~^ excessive_nesting

    loop { break {{{{1}}}} };
    //~^ excessive_nesting
    loop {{{{{{}}}}}}
    //~^ excessive_nesting

    match {{{{{{true}}}}}} {
    //~^ excessive_nesting
        true => {{{{}}}},
        //~^ excessive_nesting
        false => {{{{}}}},
        //~^ excessive_nesting
    }

    {
        {
            {
                {
                //~^ excessive_nesting
                    println!("warning! :)");
                }
            }
        }
    }
}

async fn b() -> u32 {
    async fn c() -> u32 {{{{{{{0}}}}}}}
    //~^ excessive_nesting

    c().await
}

async fn a() {
    {{{{b().await}}}};
    //~^ excessive_nesting
}
