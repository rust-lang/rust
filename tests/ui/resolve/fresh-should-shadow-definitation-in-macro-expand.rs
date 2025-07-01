//@ check-pass
//@ edition:2018

// issue#95237

fn f_without_definition_f() {
    let f = || -> i16 { 42 };
    let a: i16 = m!();
    macro_rules! m {() => ( f() )}
    use m;
}

fn f_without_closure_f() {
    fn f() -> i8 { 42 }
    let a: i8 = m!();
    macro_rules! m {() => ( f() )}
    use m;
}

fn f0() {
    let f = || -> i16 { 42 };
    fn f() -> i8 { 42 }
    let a: i16 = m!();
    macro_rules! m {() => ( f() )}
    use m;
}

fn f1() {
    let f = || -> i16 { 42 };
    let a: i16 = m!();
    fn f() -> i8 { 42 }
    macro_rules! m {() => ( f() )}
    use m;
}

fn f2() {
    let f = || -> i16 { 42 };
    let a: i16 = m!();
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> i8 { 42 }
}

fn f3() {
    let f = || -> i16 { 42 };
    macro_rules! m {() => ( f() )}
    use m;
    let a: i16 = m!();
    fn f() -> i8 { 42 }
}

fn f4() {
    let f = || -> i16 { 42 };
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> i8 { 42 }
    let a: i16 = m!();
}

fn f5() {
    let f = || -> i16 { 42 };
    fn f() -> i8 { 42 }
    macro_rules! m {() => ( f() )}
    use m;
        let a: i16 = m!();
}

fn f6() {
    fn f() -> i8 { 42 }
    let f = || -> i16 { 42 };
    let a: i16 = m!();
    macro_rules! m {() => ( f() )}
    use m;
}

fn f7() {
    fn f() -> i8 { 42 }
    let f = || -> i16 { 42 };
    macro_rules! m {() => ( f() )}
    use m;
    let a: i16 = m!();
}

fn f8() {
    fn f() -> i8 { 42 }
    let a: i8 = m!();
    let f = || -> i16 { 42 };
    macro_rules! m {() => ( f() )}
    use m;
}

fn f9() {
    fn f() -> i8 { 42 }
    let a: i8 = m!();
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> i16 { 42 };
}

fn f10() {
    fn f() -> i8 { 42 }
    macro_rules! m {() => ( f() )}
    use m;
    let a: i8 = m!();
    let f = || -> i16 { 42 };
}

fn f11() {
    fn f() -> i8 { 42 }
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> i16 { 42 };
    let a: i8 = m!();
}

fn f12() {
    let a: i8 = m!();
    fn f() -> i8 { 42 }
    let f = || -> i16 { 42 };
    macro_rules! m {() => ( f() )}
    use m;
}

fn f13() {
    let a: i8 = m!();
    let f = || -> i16 { 42 };
    fn f() -> i8 { 42 }
    macro_rules! m {() => ( f() )}
    use m;
}

fn f14() {
    let a: i8 = m!();
    let f = || -> i16 { 42 };
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> i8 { 42 }
}

fn f15() {
    let a: i8 = m!();
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> i16 { 42 };
    fn f() -> i8 { 42 }
}

fn f16() {
    let a: i8 = m!();
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> i8 { 42 }
    let f = || -> i16 { 42 };
}

fn f17() {
    let a: i8 = m!();
    fn f() -> i8 { 42 }
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> i16 { 42 };
}

fn f18() {
    macro_rules! m {() => ( f() )}
    use m;
    let a: i8 = m!();
    fn f() -> i8 { 42 }
    let f = || -> i16 { 42 };
}

fn f19() {
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> i8 { 42 }
    let a: i8 = m!();
    let f = || -> i16 { 42 };
}

fn f20() {
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> i8 { 42 }
    let f = || -> i16 { 42 };
    let a: i8 = m!();
}

fn f21() {
    macro_rules! m {() => ( f() )}
    use m;
    let a: i8 = m!();
    let f = || -> i16 { 42 };
    fn f() -> i8 { 42 }
}

fn f22() {
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> i16 { 42 };
    fn f() -> i8 { 42 }
    let a: i8 = m!();
}

fn f23() {
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> i16 { 42 };
    let a: i8 = m!();
    fn f() -> i8 { 42 }
}

fn main () {}
