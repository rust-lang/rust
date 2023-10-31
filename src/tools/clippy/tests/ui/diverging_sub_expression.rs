#![warn(clippy::diverging_sub_expression)]
#![allow(clippy::match_same_arms, clippy::overly_complex_bool_expr)]
#![allow(clippy::nonminimal_bool)]
#[allow(clippy::empty_loop)]
fn diverge() -> ! {
    loop {}
}

struct A;

impl A {
    fn foo(&self) -> ! {
        diverge()
    }
}

#[allow(unused_variables, clippy::unnecessary_operation, clippy::short_circuit_statement)]
fn main() {
    let b = true;
    b || diverge();
    b || A.foo();
}

#[allow(dead_code, unused_variables)]
#[rustfmt::skip]
fn foobar() {
    loop {
        let x = match 5 {
            4 => return,
            5 => continue,
            6 => true || return,
            7 => true || continue,
            8 => break,
            9 => diverge(),
            3 => true || diverge(),
            10 => match 42 {
                99 => return,
                _ => true || panic!("boo"),
            },
            // lint blocks as well
            15 => true || { return; },
            16 => false || { return; },
            // ... and when it's a single expression
            17 => true || { return },
            18 => false || { return },
            // ... but not when there's both an expression and a statement
            19 => true || { _ = 1; return },
            20 => false || { _ = 1; return },
            // ... or multiple statements
            21 => true || { _ = 1; return; },
            22 => false || { _ = 1; return; },
            23 => true || { return; true },
            24 => true || { return; true },
            _ => true || break,
        };
    }
}
