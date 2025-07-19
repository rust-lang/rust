//@ check-pass
//@ edition:2018
// issue#95237


#![feature(decl_macro)]

type FnF = i8;
type LetF = i16;

fn f_without_definition_f() {
    let f = || -> LetF { 42 };
    let a: LetF = m!();
    macro m() { f() }
}

fn f_without_closure_f() {
    fn f() -> FnF { 42 }
    let a: FnF = m!();
    macro m() { f() }
}

fn f0() {
    let f = || -> LetF { 42 };
    fn f() -> FnF { 42 }
    let a: LetF = m!();
    macro m() { f() }
}

fn f1() {
    let f = || -> LetF { 42 };
    let a: LetF = m!();
    fn f() -> FnF { 42 }
    macro m() { f() }
}

fn f2() {
    let f = || -> LetF { 42 };
    let a: LetF = m!();
    macro m() { f() }
    fn f() -> FnF { 42 }
}

fn f3() {
    let f = || -> LetF { 42 };
    macro m() { f() }
    let a: LetF = m!();
    fn f() -> FnF { 42 }
}

fn f4() {
    let f = || -> LetF { 42 };
    macro m() { f() }
    fn f() -> FnF { 42 }
    let a: LetF = m!();
}

fn f5() {
    let f = || -> LetF { 42 };
    fn f() -> FnF { 42 }
    macro m() { f() }
        let a: LetF = m!();
}

fn f6() {
    fn f() -> FnF { 42 }
    let f = || -> LetF { 42 };
    let a: LetF = m!();
    macro m() { f() }
}

fn f7() {
    fn f() -> FnF { 42 }
    let f = || -> LetF { 42 };
    macro m() { f() }
    let a: LetF = m!();
}

fn f8() {
    fn f() -> FnF { 42 }
    let a: FnF = m!();
    let f = || -> LetF { 42 };
    macro m() { f() }
}

fn f9() {
    fn f() -> FnF { 42 }
    let a: FnF = m!();
    macro m() { f() }
    let f = || -> LetF { 42 };
}

fn f10() {
    fn f() -> FnF { 42 }
    macro m() { f() }
    let a: FnF = m!();
    let f = || -> LetF { 42 };
}

fn f11() {
    fn f() -> FnF { 42 }
    macro m() { f() }
    let f = || -> LetF { 42 };
    let a: FnF = m!();
}

fn f12() {
    let a: FnF = m!();
    fn f() -> FnF { 42 }
    let f = || -> LetF { 42 };
    macro m() { f() }
}

fn f13() {
    let a: FnF = m!();
    let f = || -> LetF { 42 };
    fn f() -> FnF { 42 }
    macro m() { f() }
}

fn f14() {
    let a: FnF = m!();
    let f = || -> LetF { 42 };
    macro m() { f() }
    fn f() -> FnF { 42 }
}

fn f15() {
    let a: FnF = m!();
    macro m() { f() }
    let f = || -> LetF { 42 };
    fn f() -> FnF { 42 }
}

fn f16() {
    let a: FnF = m!();
    macro m() { f() }
    fn f() -> FnF { 42 }
    let f = || -> LetF { 42 };
}

fn f17() {
    let a: FnF = m!();
    fn f() -> FnF { 42 }
    macro m() { f() }
    let f = || -> LetF { 42 };
}

fn f18() {
    macro m() { f() }
    let a: FnF = m!();
    fn f() -> FnF { 42 }
    let f = || -> LetF { 42 };
}

fn f19() {
    macro m() { f() }
    fn f() -> FnF { 42 }
    let a: FnF = m!();
    let f = || -> LetF { 42 };
}

fn f20() {
    macro m() { f() }
    fn f() -> FnF { 42 }
    let f = || -> LetF { 42 };
    let a: FnF = m!();
}

fn f21() {
    macro m() { f() }
    let a: FnF = m!();
    let f = || -> LetF { 42 };
    fn f() -> FnF { 42 }
}

fn f22() {
    macro m() { f() }
    let f = || -> LetF { 42 };
    fn f() -> FnF { 42 }
    let a: FnF = m!();
}

fn f23() {
    macro m() { f() }
    let f = || -> LetF { 42 };
    let a: FnF = m!();
    fn f() -> FnF { 42 }
}

fn main () {}
