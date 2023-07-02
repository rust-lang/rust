//@aux-build:proc_macros.rs:proc-macro
#![rustfmt::skip]
#![feature(custom_inner_attributes)]
#![allow(unused)]
#![allow(clippy::let_and_return)]
#![allow(clippy::redundant_closure_call)]
#![allow(clippy::no_effect)]
#![allow(clippy::unnecessary_operation)]
#![allow(clippy::never_loop)]
#![allow(clippy::needless_if)]
#![warn(clippy::excessive_nesting)]
#![allow(clippy::collapsible_if)]

#[macro_use]
extern crate proc_macros;

static X: u32 = {
    let x = {
        let y = {
            let z = {
                let w = { 3 };
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
    a.a({{{{{{{{{0}}}}}}}}});
    (0, {{{{{{{1}}}}}}});

    if true {
        if true {
            if true {
                if true {
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
    let z = y + {{{{{{{{{5}}}}}}}}};
    [0, {{{{{{{{{{0}}}}}}}}}}];
    let mut xx = [0; {{{{{{{{100}}}}}}}}];
    xx[{{{{{{{{{{{{{{{{{{{{{{{{3}}}}}}}}}}}}}}}}}}}}}}}}];
    &mut {{{{{{{{{{y}}}}}}}}}};

    for i in {{{{xx}}}} {{{{{{{{}}}}}}}}

    while let Some(i) = {{{{{{Some(1)}}}}}} {{{{{{{}}}}}}}
    
    while {{{{{{{{true}}}}}}}} {{{{{{{{{}}}}}}}}}

    let d = D { d: {{{{{{{{{{{{{{{{{{{{{{{3}}}}}}}}}}}}}}}}}}}}}}} };

    {{{{1;}}}}..{{{{{{3}}}}}};
    {{{{1;}}}}..={{{{{{{{{{{{{{{{{{{{{{{{{{6}}}}}}}}}}}}}}}}}}}}}}}}}};
    ..{{{{{{{5}}}}}}};
    ..={{{{{3}}}}};
    {{{{{1;}}}}}..;

    loop { break {{{{1}}}} };
    loop {{{{{{}}}}}}

    match {{{{{{true}}}}}} {
        true => {{{{}}}},
        false => {{{{}}}},
    }

    {
        {
            {
                {
                    println!("warning! :)");
                }
            }
        }
    }
}

async fn b() -> u32 {
    async fn c() -> u32 {{{{{{{0}}}}}}}

    c().await
}

async fn a() {
    {{{{b().await}}}};
}
