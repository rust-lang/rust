//@ check-pass
//@ edition:2018
// issue#95237


#![feature(decl_macro)]

fn f_without_definition_f() {
    let f = || -> i16 { 42 };
    let a: i16 = m!();
    macro m() { f() }
}

fn f_without_closure_f() {
    fn f() -> i8 { 42 }
    let a: i8 = m!();
    macro m() { f() }
}

fn f0() {
    let f = || -> i16 { 42 };
    fn f() -> i8 { 42 }
    let a: i16 = m!();
    macro m() { f() }
}

fn f1() {
    let f = || -> i16 { 42 };
    let a: i16 = m!();
    fn f() -> i8 { 42 }
    macro m() { f() }
}

fn f2() {
    let f = || -> i16 { 42 };
    let a: i16 = m!();
    macro m() { f() }
    fn f() -> i8 { 42 }
}

fn f3() {
    let f = || -> i16 { 42 };
    macro m() { f() }
    let a: i16 = m!();
    fn f() -> i8 { 42 }
}

fn f4() {
    let f = || -> i16 { 42 };
    macro m() { f() }
    fn f() -> i8 { 42 }
    let a: i16 = m!();
}

fn f5() {
    let f = || -> i16 { 42 };
    fn f() -> i8 { 42 }
    macro m() { f() }
        let a: i16 = m!();
}

fn f6() {
    fn f() -> i8 { 42 }
    let f = || -> i16 { 42 };
    let a: i16 = m!();
    macro m() { f() }
}

fn f7() {
    fn f() -> i8 { 42 }
    let f = || -> i16 { 42 };
    macro m() { f() }
    let a: i16 = m!();
}

fn f8() {
    fn f() -> i8 { 42 }
    let a: i8 = m!();
    let f = || -> i16 { 42 };
    macro m() { f() }
}

fn f9() {
    fn f() -> i8 { 42 }
    let a: i8 = m!();
    macro m() { f() }
    let f = || -> i16 { 42 };
}

fn f10() {
    fn f() -> i8 { 42 }
    macro m() { f() }
    let a: i8 = m!();
    let f = || -> i16 { 42 };
}

fn f11() {
    fn f() -> i8 { 42 }
    macro m() { f() }
    let f = || -> i16 { 42 };
    let a: i8 = m!();
}

fn f12() {
    let a: i8 = m!();
    fn f() -> i8 { 42 }
    let f = || -> i16 { 42 };
    macro m() { f() }
}

fn f13() {
    let a: i8 = m!();
    let f = || -> i16 { 42 };
    fn f() -> i8 { 42 }
    macro m() { f() }
}

fn f14() {
    let a: i8 = m!();
    let f = || -> i16 { 42 };
    macro m() { f() }
    fn f() -> i8 { 42 }
}

fn f15() {
    let a: i8 = m!();
    macro m() { f() }
    let f = || -> i16 { 42 };
    fn f() -> i8 { 42 }
}

fn f16() {
    let a: i8 = m!();
    macro m() { f() }
    fn f() -> i8 { 42 }
    let f = || -> i16 { 42 };
}

fn f17() {
    let a: i8 = m!();
    fn f() -> i8 { 42 }
    macro m() { f() }
    let f = || -> i16 { 42 };
}

fn f18() {
    macro m() { f() }
    let a: i8 = m!();
    fn f() -> i8 { 42 }
    let f = || -> i16 { 42 };
}

fn f19() {
    macro m() { f() }
    fn f() -> i8 { 42 }
    let a: i8 = m!();
    let f = || -> i16 { 42 };
}

fn f20() {
    macro m() { f() }
    fn f() -> i8 { 42 }
    let f = || -> i16 { 42 };
    let a: i8 = m!();
}

fn f21() {
    macro m() { f() }
    let a: i8 = m!();
    let f = || -> i16 { 42 };
    fn f() -> i8 { 42 }
}

fn f22() {
    macro m() { f() }
    let f = || -> i16 { 42 };
    fn f() -> i8 { 42 }
    let a: i8 = m!();
}

fn f23() {
    macro m() { f() }
    let f = || -> i16 { 42 };
    let a: i8 = m!();
    fn f() -> i8 { 42 }
}

fn main () {}
