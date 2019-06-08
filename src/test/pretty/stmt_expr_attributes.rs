// pp-exact

#![feature(box_syntax)]
#![feature(rustc_attrs)]
#![feature(stmt_expr_attributes)]

fn main() { }

fn _0() {

    #[rustc_dummy]
    foo();
}

fn _1() {

    #[rustc_dummy]
    unsafe {
        // code
    }
}

fn _2() {

    #[rustc_dummy]
    { foo(); }

    {
        #![rustc_dummy]

        foo()
    }
}

fn _3() {

    #[rustc_dummy]
    match () { _ => { } }
}

fn _4() {

    #[rustc_dummy]
    match () {
        #![rustc_dummy]
        _ => (),
    }

    let _ =
        #[rustc_dummy] match () {
                           #![rustc_dummy]
                           () => (),
                       };
}

fn _5() {

    #[rustc_dummy]
    let x = 1;

    let x = #[rustc_dummy] 1;

    let y = ();
    let z = ();

    foo3(x, #[rustc_dummy] y, z);

    qux(3 + #[rustc_dummy] 2);
}

fn _6() {

    #[rustc_dummy]
    [#![rustc_dummy] 1, 2, 3];

    let _ = #[rustc_dummy] [#![rustc_dummy] 1, 2, 3];

    #[rustc_dummy]
    [#![rustc_dummy] 1; 4];

    let _ = #[rustc_dummy] [#![rustc_dummy] 1; 4];
}

struct Foo {
    data: (),
}

struct Bar(());

fn _7() {

    #[rustc_dummy]
    Foo{#![rustc_dummy] data: (),};

    let _ = #[rustc_dummy] Foo{#![rustc_dummy] data: (),};
}

fn _8() {

    #[rustc_dummy]
    (#![rustc_dummy] );

    #[rustc_dummy]
    (#![rustc_dummy] 0);

    #[rustc_dummy]
    (#![rustc_dummy] 0,);

    #[rustc_dummy]
    (#![rustc_dummy] 0, 1);
}

fn _9() {
    macro_rules! stmt_mac((  ) => { let _ = (  ) ; });

    #[rustc_dummy]
    stmt_mac!();

    #[rustc_dummy]
    stmt_mac!{ };

    #[rustc_dummy]
    stmt_mac![];

    #[rustc_dummy]
    stmt_mac!{ }

    let _ = ();
}

macro_rules! expr_mac((  ) => { (  ) });

fn _10() {
    let _ = #[rustc_dummy] expr_mac!();
    let _ = #[rustc_dummy] expr_mac![];
    let _ = #[rustc_dummy] expr_mac!{ };
}

fn _11() {
    let _ = #[rustc_dummy] box 0;
    let _: [(); 0] = #[rustc_dummy] [#![rustc_dummy] ];
    let _ = #[rustc_dummy] [#![rustc_dummy] 0, 0];
    let _ = #[rustc_dummy] [#![rustc_dummy] 0; 0];
    let _ = #[rustc_dummy] foo();
    let _ = #[rustc_dummy] 1i32.clone();
    let _ = #[rustc_dummy] (#![rustc_dummy] );
    let _ = #[rustc_dummy] (#![rustc_dummy] 0);
    let _ = #[rustc_dummy] (#![rustc_dummy] 0,);
    let _ = #[rustc_dummy] (#![rustc_dummy] 0, 0);
    let _ = #[rustc_dummy] 0 + #[rustc_dummy] 0;
    let _ = #[rustc_dummy] !0;
    let _ = #[rustc_dummy] -0i32;
    let _ = #[rustc_dummy] false;
    let _ = #[rustc_dummy] 'c';
    let _ = #[rustc_dummy] 0;
    let _ = #[rustc_dummy] 0 as usize;
    let _ =
        #[rustc_dummy] while false {
                           #![rustc_dummy]
                       };
    let _ =
        #[rustc_dummy] while let None = Some(()) {
                           #![rustc_dummy]
                       };
    let _ =
        #[rustc_dummy] for _ in 0..0 {
                           #![rustc_dummy]
                       };
    // FIXME: pp bug, two spaces after the loop
    let _ =
        #[rustc_dummy] loop  {
                           #![rustc_dummy]
                       };
    let _ =
        #[rustc_dummy] match false {
                           #![rustc_dummy]
                           _ => (),
                       };
    let _ = #[rustc_dummy] || #[rustc_dummy] ();
    let _ = #[rustc_dummy] move || #[rustc_dummy] ();
    let _ =
        #[rustc_dummy] ||
                           {
                               #![rustc_dummy]
                               #[rustc_dummy]
                               ()
                           };
    let _ =
        #[rustc_dummy] move ||
                           {
                               #![rustc_dummy]
                               #[rustc_dummy]
                               ()
                           };
    let _ =
        #[rustc_dummy] {
                           #![rustc_dummy]
                       };
    let _ =
        #[rustc_dummy] {
                           #![rustc_dummy]
                           let _ = ();
                       };
    let _ =
        #[rustc_dummy] {
                           #![rustc_dummy]
                           let _ = ();
                           ()
                       };
    let mut x = 0;
    let _ = #[rustc_dummy] x = 15;
    let _ = #[rustc_dummy] x += 15;
    let s = Foo{data: (),};
    let _ = #[rustc_dummy] s.data;
    let _ = (#[rustc_dummy] s).data;
    let t = Bar(());
    let _ = #[rustc_dummy] t.0;
    let _ = (#[rustc_dummy] t).0;
    let v = vec!(0);
    let _ = #[rustc_dummy] v[0];
    let _ = (#[rustc_dummy] v)[0];
    let _ = #[rustc_dummy] 0..#[rustc_dummy] 0;
    let _ = #[rustc_dummy] 0..;
    let _ = #[rustc_dummy] (0..0);
    let _ = #[rustc_dummy] (0..);
    let _ = #[rustc_dummy] (..0);
    let _ = #[rustc_dummy] (..);
    let _: fn(&u32) -> u32 = #[rustc_dummy] std::clone::Clone::clone;
    let _ = #[rustc_dummy] &0;
    let _ = #[rustc_dummy] &mut 0;
    let _ = #[rustc_dummy] &#[rustc_dummy] 0;
    let _ = #[rustc_dummy] &mut #[rustc_dummy] 0;
    // FIXME: pp bug, extra space after keyword?
    while false { let _ = #[rustc_dummy] continue ; }
    while true { let _ = #[rustc_dummy] break ; }
    || #[rustc_dummy] return;
    let _ = #[rustc_dummy] expr_mac!();
    let _ = #[rustc_dummy] expr_mac![];
    let _ = #[rustc_dummy] expr_mac!{ };
    let _ = #[rustc_dummy] Foo{#![rustc_dummy] data: (),};
    let _ = #[rustc_dummy] Foo{#![rustc_dummy] ..s};
    let _ = #[rustc_dummy] Foo{#![rustc_dummy] data: (), ..s};
    let _ = #[rustc_dummy] (#![rustc_dummy] 0);
}

fn _12() {
    #[rustc_dummy]
    let _ = 0;

    #[rustc_dummy]
    0;

    #[rustc_dummy]
    expr_mac!();

    #[rustc_dummy]
    {
        #![rustc_dummy]
    }
}

/////////////////

fn foo() { }
fn foo3(_: i32, _: (), _: ()) { }
fn qux(_: i32) { }
