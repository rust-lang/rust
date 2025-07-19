//@ check-pass
//@ edition:2018

// issue#95237

type FnF = i8;
type LetF = i16;

fn b_without_definition_f() {
    let f = || -> LetF { 42 };
    {
        let a: LetF = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
}

fn b_without_closure_f() {
    fn f() -> FnF { 42 }
    {
        let a: FnF = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
}

fn b0() {
    let f = || -> LetF { 42 };
    fn f() -> FnF { 42 }
    {
        let a: LetF = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
}

fn b1() {
    let f = || -> LetF { 42 };
    {
        let a: LetF = m!();
    }
    fn f() -> FnF { 42 }
    macro_rules! m {() => ( f() )}
    use m;
}

fn b2() {
    let f = || -> LetF { 42 };
    {
        let a: LetF = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> FnF { 42 }
}

fn b3() {
    let f = || -> LetF { 42 };
    macro_rules! m {() => ( f() )}
    use m;
    {
        let a: LetF = m!();
    }
    fn f() -> FnF { 42 }
}

fn b4() {
    let f = || -> LetF { 42 };
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> FnF { 42 }
    {
        let a: LetF = m!();
    }
}

fn b5() {
    let f = || -> LetF { 42 };
    fn f() -> FnF { 42 }
    macro_rules! m {() => ( f() )}
    use m;
    {
        let a: LetF = m!();
    }
}

fn b6() {
    fn f() -> FnF { 42 }
    let f = || -> LetF { 42 };
    {
        let a: LetF = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
}

fn b7() {
    fn f() -> FnF { 42 }
    let f = || -> LetF { 42 };
    macro_rules! m {() => ( f() )}
    use m;
    {
        let a: LetF = m!();
    }
}

fn b8() {
    fn f() -> FnF { 42 }
    {
        let a: FnF = m!();
    }
    let f = || -> LetF { 42 };
    macro_rules! m {() => ( f() )}
    use m;
}

fn b9() {
    fn f() -> FnF { 42 }
    {
        let a: FnF = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> LetF { 42 };
}

fn b10() {
    fn f() -> FnF { 42 }
    macro_rules! m {() => ( f() )}
    use m;
    {
        let a: FnF = m!();
    }
    let f = || -> LetF { 42 };
}

fn b11() {
    fn f() -> FnF { 42 }
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> LetF { 42 };
    {
        let a: FnF = m!();
    }
}

fn b12() {
    {
        let a: FnF = m!();
    }
    fn f() -> FnF { 42 }
    let f = || -> LetF { 42 };
    macro_rules! m {() => ( f() )}
    use m;
}

fn b13() {
    {
        let a: FnF = m!();
    }
    let f = || -> LetF { 42 };
    fn f() -> FnF { 42 }
    macro_rules! m {() => ( f() )}
    use m;
}

fn b14() {
    {
        let a: FnF = m!();
    }
    let f = || -> LetF { 42 };
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> FnF { 42 }
}

fn b15() {
    {
        let a: FnF = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> LetF { 42 };
    fn f() -> FnF { 42 }
}

fn b16() {
    {
        let a: FnF = m!();
    }
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> FnF { 42 }
    let f = || -> LetF { 42 };
}

fn b17() {
    {
        let a: FnF = m!();
    }
    fn f() -> FnF { 42 }
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> LetF { 42 };
}

fn b18() {
    macro_rules! m {() => ( f() )}
    use m;
    {
        let a: FnF = m!();
    }
    fn f() -> FnF { 42 }
    let f = || -> LetF { 42 };
}

fn b19() {
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> FnF { 42 }
    {
        let a: FnF = m!();
    }
    let f = || -> LetF { 42 };
}

fn b20() {
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> FnF { 42 }
    let f = || -> LetF { 42 };
    {
        let a: FnF = m!();
    }
}

fn b21() {
    macro_rules! m {() => ( f() )}
    use m;
    {
        let a: FnF = m!();
    }
    let f = || -> LetF { 42 };
    fn f() -> FnF { 42 }
}

fn b22() {
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> LetF { 42 };
    fn f() -> FnF { 42 }
    {
        let a: FnF = m!();
    }
}

fn b23() {
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> LetF { 42 };
    {
        let a: FnF = m!();
    }
    fn f() -> FnF { 42 }
}

fn main () {}
