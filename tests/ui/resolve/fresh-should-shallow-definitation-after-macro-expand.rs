//@ check-pass
//@ edition:2018

// issue#95237

fn f0() {
    fn f() -> i8 { 42 }
    let f = || -> i16 { 42 };

    let a: i16 = m!();
    macro_rules! m {() => ( f() )}
    use m;
    let b: i16 = m!();
}

fn f1() {
    fn f() -> i8 { 42 }

    let a: i8 = m!();
    let f = || -> i16 { 42 };
    macro_rules! m {() => ( f() )}
    use m;
    let b: i16 = m!();
}

fn f2() {
    fn f() -> i8 { 42 }

    let a: i8 = m!();
    macro_rules! m {() => ( f() )}
    use m;
    let b: i8 = m!();

    let f = || -> i16 { 42 };
}

fn f3() {
    let f = || -> i16 { 42 };

    let a: i16 = m!();
    macro_rules! m {() => ( f() )}
    use m;
    let b: i16 = m!();
}

fn main () {}
