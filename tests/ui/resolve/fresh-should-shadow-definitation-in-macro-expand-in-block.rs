//@ check-pass
//@ edition:2018

// issue#95237

fn b_without_definition_f() {
    let f = || -> i16 { 42 };
    {
        let a: i16 = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
}

fn b_without_closure_f() {
    fn f() -> i8 { 42 }
    {
        let a: i8 = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
}

fn b0() {
    let f = || -> i16 { 42 };
    fn f() -> i8 { 42 }
    {
        let a: i16 = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
}

fn b1() {
    let f = || -> i16 { 42 };
    {
        let a: i16 = m!();
    }
    fn f() -> i8 { 42 }
    macro_rules! m {() => ( f() )}
    use m;
}

fn b2() {
    let f = || -> i16 { 42 };
    {
        let a: i16 = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> i8 { 42 }
}

fn b3() {
    let f = || -> i16 { 42 };
    macro_rules! m {() => ( f() )}
    use m;
    {
        let a: i16 = m!();
    }
    fn f() -> i8 { 42 }
}

fn b4() {
    let f = || -> i16 { 42 };
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> i8 { 42 }
    {
        let a: i16 = m!();
    }
}

fn b5() {
    let f = || -> i16 { 42 };
    fn f() -> i8 { 42 }
    macro_rules! m {() => ( f() )}
    use m;
    {
        let a: i16 = m!();
    }
}

fn b6() {
    fn f() -> i8 { 42 }
    let f = || -> i16 { 42 };
    {
        let a: i16 = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
}

fn b7() {
    fn f() -> i8 { 42 }
    let f = || -> i16 { 42 };
    macro_rules! m {() => ( f() )}
    use m;
    {
        let a: i16 = m!();
    }
}

fn b8() {
    fn f() -> i8 { 42 }
    {
        let a: i8 = m!();
    }
    let f = || -> i16 { 42 };
    macro_rules! m {() => ( f() )}
    use m;
}

fn b9() {
    fn f() -> i8 { 42 }
    {
        let a: i8 = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> i16 { 42 };
}

fn b10() {
    fn f() -> i8 { 42 }
    macro_rules! m {() => ( f() )}
    use m;
    {
        let a: i8 = m!();
    }
    let f = || -> i16 { 42 };
}

fn b11() {
    fn f() -> i8 { 42 }
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> i16 { 42 };
    {
        let a: i8 = m!();
    }
}

fn b12() {
    {
        let a: i8 = m!();
    }
    fn f() -> i8 { 42 }
    let f = || -> i16 { 42 };
    macro_rules! m {() => ( f() )}
    use m;
}

fn b13() {
    {
        let a: i8 = m!();
    }
    let f = || -> i16 { 42 };
    fn f() -> i8 { 42 }
    macro_rules! m {() => ( f() )}
    use m;
}

fn b14() {
    {
        let a: i8 = m!();
    }
    let f = || -> i16 { 42 };
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> i8 { 42 }
}

fn b15() {
    {
        let a: i8 = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> i16 { 42 };
    fn f() -> i8 { 42 }
}

fn b16() {
    {
        let a: i8 = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> i8 { 42 }
    let f = || -> i16 { 42 };
}

fn b17() {
    {
        let a: i8 = m!();
    }
    fn f() -> i8 { 42 }
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> i16 { 42 };
}

fn b18() {
    macro_rules! m {() => ( f() )}
    use m;
    {
        let a: i8 = m!();
    }
    fn f() -> i8 { 42 }
    let f = || -> i16 { 42 };
}

fn b19() {
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> i8 { 42 }
    {
        let a: i8 = m!();
    }
    let f = || -> i16 { 42 };
}

fn b20() {
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> i8 { 42 }
    let f = || -> i16 { 42 };
    {
        let a: i8 = m!();
    }
}

fn b21() {
    macro_rules! m {() => ( f() )}
    use m;
    {
        let a: i8 = m!();
    }
    let f = || -> i16 { 42 };
    fn f() -> i8 { 42 }
}

fn b22() {
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> i16 { 42 };
    fn f() -> i8 { 42 }
    {
        let a: i8 = m!();
    }
}

fn b23() {
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> i16 { 42 };
    {
        let a: i8 = m!();
    }
    fn f() -> i8 { 42 }
}

fn main () {}
