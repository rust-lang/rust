// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pp-exact

#![feature(custom_attribute)]
#![feature(box_syntax)]
#![feature(placement_in_syntax)]
#![feature(stmt_expr_attributes)]

fn main() { }

fn _0() {

    #[attr]
    foo();
}

fn _1() {

    #[attr]
    unsafe {
        // code
    }
}

fn _2() {

    #[attr]
    { foo(); }

    {
        #![attr]

        foo()
    }
}

fn _3() {

    #[attr]
    match () { _ => { } }
}

fn _4() {

    #[attr]
    match () {
        #![attr]
        _ => (),
    }

    let _ =
        #[attr] match () {
                    #![attr]
                    () => (),
                };
}

fn _5() {

    #[attr]
    let x = 1;

    let x = #[attr] 1;

    let y = ();
    let z = ();

    foo3(x, #[attr] y, z);

    qux(3 + #[attr] 2);
}

fn _6() {

    #[attr]
    [#![attr] 1, 2, 3];

    let _ = #[attr] [#![attr] 1, 2, 3];

    #[attr]
    [#![attr] 1; 4];

    let _ = #[attr] [#![attr] 1; 4];
}

struct Foo {
    data: (),
}

struct Bar(());

fn _7() {

    #[attr]
    Foo{#![attr] data: (),};

    let _ = #[attr] Foo{#![attr] data: (),};
}

fn _8() {

    #[attr]
    (#![attr] );

    #[attr]
    (#![attr] 0);

    #[attr]
    (#![attr] 0,);

    #[attr]
    (#![attr] 0, 1);
}

fn _9() {
    macro_rules! stmt_mac((  ) => { let _ = (  ) ; });

    #[attr]
    stmt_mac!();

    /*
    // pre existing pp bug: delimiter styles gets lost:

    #[attr]
    stmt_mac!{ };

    #[attr]
    stmt_mac![];

    #[attr]
    stmt_mac!{ } // pre-existing pp bug: compiler ICEs with a None unwrap
    */

    let _ = ();
}

macro_rules! expr_mac((  ) => { (  ) });

fn _10() {

    let _ = #[attr] expr_mac!();

    /*
    // pre existing pp bug: delimiter styles gets lost:
    let _ = #[attr] expr_mac![];
    let _ = #[attr] expr_mac!{};
    */
}

fn _11() {
    let _ = #[attr] box 0;
    let _: [(); 0] = #[attr] [#![attr] ];
    let _ = #[attr] [#![attr] 0, 0];
    let _ = #[attr] [#![attr] 0; 0];
    let _ = #[attr] foo();
    let _ = #[attr] 1i32.clone();
    let _ = #[attr] (#![attr] );
    let _ = #[attr] (#![attr] 0);
    let _ = #[attr] (#![attr] 0,);
    let _ = #[attr] (#![attr] 0, 0);
    let _ = #[attr] 0 + #[attr] 0;
    let _ = #[attr] !0;
    let _ = #[attr] -0i32;
    let _ = #[attr] false;
    let _ = #[attr] 'c';
    let _ = #[attr] 0;
    let _ = #[attr] 0 as usize;
    let _ =
        #[attr] while false {
                    #![attr]
                };
    let _ =
        #[attr] while let None = Some(()) {
                    #![attr]
                };
    let _ =
        #[attr] for _ in 0..0 {
                    #![attr]
                };
    // FIXME: pp bug, two spaces after the loop
    let _ =
        #[attr] loop  {
                    #![attr]
                };
    let _ =
        #[attr] match false {
                    #![attr]
                    _ => (),
                };
    let _ = #[attr] || #[attr] ();
    let _ = #[attr] move || #[attr] ();
    let _ =
        #[attr] ||
                    {
                        #![attr]
                        #[attr]
                        ()
                    };
    let _ =
        #[attr] move ||
                    {
                        #![attr]
                        #[attr]
                        ()
                    };
    let _ =
        #[attr] {
                    #![attr]
                };
    let _ =
        #[attr] {
                    #![attr]
                    let _ = ();
                };
    let _ =
        #[attr] {
                    #![attr]
                    let _ = ();
                    ()
                };
    let mut x = 0;
    let _ = #[attr] x = 15;
    let _ = #[attr] x += 15;
    let s = Foo{data: (),};
    let _ = #[attr] s.data;
    let _ = (#[attr] s).data;
    let t = Bar(());
    let _ = #[attr] t.0;
    let _ = (#[attr] t).0;
    let v = vec!(0);
    let _ = #[attr] v[0];
    let _ = (#[attr] v)[0];
    let _ = #[attr] 0..#[attr] 0;
    let _ = #[attr] 0..;
    let _ = #[attr] (0..0);
    let _ = #[attr] (0..);
    let _ = #[attr] (..0);
    let _ = #[attr] (..);
    let _: fn(&u32) -> u32 = #[attr] std::clone::Clone::clone;
    let _ = #[attr] &0;
    let _ = #[attr] &mut 0;
    let _ = #[attr] &#[attr] 0;
    let _ = #[attr] &mut #[attr] 0;
    // FIXME: pp bug, extra space after keyword?
    while false { let _ = #[attr] continue ; }
    while true { let _ = #[attr] break ; }
    || #[attr] return;
    let _ = #[attr] expr_mac!();
    /* FIXME: pp bug, loosing delimiter styles
    let _ = #[attr] expr_mac![];
    let _ = #[attr] expr_mac!{};
    */
    let _ = #[attr] Foo{#![attr] data: (),};
    let _ = #[attr] Foo{#![attr] ..s};
    let _ = #[attr] Foo{#![attr] data: (), ..s};
    let _ = #[attr] (#![attr] 0);
}

fn _12() {
    #[attr]
    let _ = 0;

    #[attr]
    0;

    #[attr]
    expr_mac!();

    #[attr]
    {
        #![attr]
    }
}

/////////////////

fn foo() { }
fn foo3(_: i32, _: (), _: ()) { }
fn qux(_: i32) { }
