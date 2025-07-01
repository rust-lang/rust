//@ check-pass
//@ edition:2018

// issue#95237

type FnF = i8;
type LetF = i16;

fn f_without_definition_f() {
    let f = || -> LetF { 42 };
    let a: LetF = m!();
    macro_rules! m {() => ( f() )}
    use m;
}

fn f_without_closure_f() {
    fn f() -> FnF { 42 }
    let a: FnF = m!();
    macro_rules! m {() => ( f() )}
    use m;
}

fn f0() {
    let f = || -> LetF { 42 };
    fn f() -> FnF { 42 }
    let a: LetF = m!();
    macro_rules! m {() => ( f() )}
    use m;
}

fn f1() {
    let f = || -> LetF { 42 };
    let a: LetF = m!();
    fn f() -> FnF { 42 }
    macro_rules! m {() => ( f() )}
    use m;
}

fn f2() {
    let f = || -> LetF { 42 };
    let a: LetF = m!();
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> FnF { 42 }
}

fn f3() {
    let f = || -> LetF { 42 };
    macro_rules! m {() => ( f() )}
    use m;
    let a: LetF = m!();
    fn f() -> FnF { 42 }
}

fn f4() {
    let f = || -> LetF { 42 };
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> FnF { 42 }
    let a: LetF = m!();
}

fn f5() {
    let f = || -> LetF { 42 };
    fn f() -> FnF { 42 }
    macro_rules! m {() => ( f() )}
    use m;
        let a: LetF = m!();
}

fn f6() {
    fn f() -> FnF { 42 }
    let f = || -> LetF { 42 };
    let a: LetF = m!();
    macro_rules! m {() => ( f() )}
    use m;
}

fn f7() {
    fn f() -> FnF { 42 }
    let f = || -> LetF { 42 };
    macro_rules! m {() => ( f() )}
    use m;
    let a: LetF = m!();
}

fn f8() {
    fn f() -> FnF { 42 }
    let a: FnF = m!();
    let f = || -> LetF { 42 };
    macro_rules! m {() => ( f() )}
    use m;
}

fn f9() {
    fn f() -> FnF { 42 }
    let a: FnF = m!();
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> LetF { 42 };
}

fn f10() {
    fn f() -> FnF { 42 }
    macro_rules! m {() => ( f() )}
    use m;
    let a: FnF = m!();
    let f = || -> LetF { 42 };
}

fn f11() {
    fn f() -> FnF { 42 }
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> LetF { 42 };
    let a: FnF = m!();
}

fn f12() {
    let a: FnF = m!();
    fn f() -> FnF { 42 }
    let f = || -> LetF { 42 };
    macro_rules! m {() => ( f() )}
    use m;
}

fn f13() {
    let a: FnF = m!();
    let f = || -> LetF { 42 };
    fn f() -> FnF { 42 }
    macro_rules! m {() => ( f() )}
    use m;
}

fn f14() {
    let a: FnF = m!();
    let f = || -> LetF { 42 };
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> FnF { 42 }
}

fn f15() {
    let a: FnF = m!();
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> LetF { 42 };
    fn f() -> FnF { 42 }
}

fn f16() {
    let a: FnF = m!();
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> FnF { 42 }
    let f = || -> LetF { 42 };
}

fn f17() {
    let a: FnF = m!();
    fn f() -> FnF { 42 }
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> LetF { 42 };
}

fn f18() {
    macro_rules! m {() => ( f() )}
    use m;
    let a: FnF = m!();
    fn f() -> FnF { 42 }
    let f = || -> LetF { 42 };
}

fn f19() {
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> FnF { 42 }
    let a: FnF = m!();
    let f = || -> LetF { 42 };
}

fn f20() {
    macro_rules! m {() => ( f() )}
    use m;
    fn f() -> FnF { 42 }
    let f = || -> LetF { 42 };
    let a: FnF = m!();
}

fn f21() {
    macro_rules! m {() => ( f() )}
    use m;
    let a: FnF = m!();
    let f = || -> LetF { 42 };
    fn f() -> FnF { 42 }
}

fn f22() {
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> LetF { 42 };
    fn f() -> FnF { 42 }
    let a: FnF = m!();
}

fn f23() {
    macro_rules! m {() => ( f() )}
    use m;
    let f = || -> LetF { 42 };
    let a: FnF = m!();
    fn f() -> FnF { 42 }
}

fn main () {}
