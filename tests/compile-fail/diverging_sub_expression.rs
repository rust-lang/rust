#![feature(plugin, never_type)]
#![plugin(clippy)]
#![deny(diverging_sub_expression)]

#[allow(empty_loop)]
fn diverge() -> ! { loop {} }

struct A;

impl A {
    fn foo(&self) -> ! { diverge() }
}

#[allow(unused_variables, unnecessary_operation)]
fn main() {
    let b = true;
    b || diverge(); //~ ERROR sub-expression diverges
    b || A.foo(); //~ ERROR sub-expression diverges
    let y = (5, diverge(), 6); //~ ERROR sub-expression diverges
    println!("{}", y.1);
}

#[allow(dead_code, unused_variables)]
fn foobar() {
    loop {
        let x = match 5 {
            4 => return,
            5 => continue,
            6 => (println!("foo"), return), //~ ERROR sub-expression diverges
            7 => (println!("bar"), continue), //~ ERROR sub-expression diverges
            8 => break,
            9 => diverge(),
            3 => (println!("moo"), diverge()), //~ ERROR sub-expression diverges
            10 => match 42 {
                99 => return,
                _ => ((), panic!("boo")),
            },
            _ => (println!("boo"), break), //~ ERROR sub-expression diverges
        };
    }
}
