#![feature(plugin, never_type)]
#![plugin(clippy)]
#![deny(diverging_sub_expression)]
#![allow(match_same_arms, logic_bug)]

#[allow(empty_loop)]
fn diverge() -> ! { loop {} }

struct A;

impl A {
    fn foo(&self) -> ! { diverge() }
}

#[allow(unused_variables, unnecessary_operation, short_circuit_statement)]
fn main() {
    let b = true;
    b || diverge(); //~ ERROR sub-expression diverges
    b || A.foo(); //~ ERROR sub-expression diverges
}

#[allow(dead_code, unused_variables)]
fn foobar() {
    loop {
        let x = match 5 {
            4 => return,
            5 => continue,
            6 => true || return, //~ ERROR sub-expression diverges
            7 => true || continue, //~ ERROR sub-expression diverges
            8 => break,
            9 => diverge(),
            3 => true || diverge(), //~ ERROR sub-expression diverges
            10 => match 42 {
                99 => return,
                _ => true || panic!("boo"),
            },
            _ => true || break, //~ ERROR sub-expression diverges
        };
    }
}
