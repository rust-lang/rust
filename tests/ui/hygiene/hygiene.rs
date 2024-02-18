//@ run-pass
#![allow(unused)]

fn f() {
    let x = 0;
    macro_rules! foo { () => {
        assert_eq!(x, 0);
    } }

    let x = 1;
    foo!();
}

fn g() {
    let x = 0;
    macro_rules! m { ($m1:ident, $m2:ident, $x:ident) => {
        macro_rules! $m1 { () => { ($x, x) } }
        let x = 1;
        macro_rules! $m2 { () => { ($x, x) } }
    } }

    let x = 2;
    m!(m2, m3, x);

    let x = 3;
    assert_eq!(m2!(), (2, 0));
    assert_eq!(m3!(), (2, 1));

    let x = 4;
    m!(m4, m5, x);
    assert_eq!(m4!(), (4, 0));
    assert_eq!(m5!(), (4, 1));
}

mod foo {
    macro_rules! m {
        ($f:ident : |$x:ident| $e:expr) => {
            pub fn $f() -> (i32, i32) {
                let x = 0;
                let $x = 1;
                (x, $e)
            }
        }
    }

    m!(f: |x| x + 10);
}

fn interpolated_pattern() {
    let x = 0;
    macro_rules! m {
        ($p:pat, $e:expr) => {
            let $p = 1;
            assert_eq!((x, $e), (0, 1));
        }
    }

    m!(x, x);
}

fn patterns_in_macro_generated_macros() {
    let x = 0;
    macro_rules! m {
        ($a:expr, $b:expr) => {
            assert_eq!(x, 0);
            let x = $a;
            macro_rules! n {
                () => {
                    (x, $b)
                }
            }
        }
    }

    let x = 1;
    m!(2, x);

    let x = 3;
    assert_eq!(n!(), (2, 1));
}

fn match_hygiene() {
    let x = 0;

    macro_rules! m {
        ($p:pat, $e:expr) => {
            for result in &[Ok(1), Err(1)] {
                match *result {
                    $p => { assert_eq!(($e, x), (1, 0)); }
                    Err(x) => { assert_eq!(($e, x), (2, 1)); }
                }
            }
        }
    }

    let x = 2;
    m!(Ok(x), x);
}

fn label_hygiene() {
    'a: loop {
        macro_rules! m { () => { break 'a; } }
        m!();
    }
}

fn main() {
    f();
    g();
    assert_eq!(foo::f(), (0, 11));
    interpolated_pattern();
    patterns_in_macro_generated_macros();
    match_hygiene();
}
