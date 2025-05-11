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
    //~^ diverging_sub_expression

    b || A.foo();
    //~^ diverging_sub_expression
}

#[allow(dead_code, unused_variables)]
#[rustfmt::skip]
fn foobar() {
    loop {
        let x = match 5 {
            4 => return,
            5 => continue,
            6 => true || return,
            //~^ diverging_sub_expression

            7 => true || continue,
            //~^ diverging_sub_expression

            8 => break,
            9 => diverge(),
            3 => true || diverge(),
            //~^ diverging_sub_expression

            10 => match 42 {
                99 => return,
                _ => true || panic!("boo"),
                //~^ diverging_sub_expression

            },
            // lint blocks as well
            15 => true || { return; },
            //~^ diverging_sub_expression

            16 => false || { return; },
            //~^ diverging_sub_expression

            // ... and when it's a single expression
            17 => true || { return },
            //~^ diverging_sub_expression

            18 => false || { return },
            //~^ diverging_sub_expression

            // ... but not when there's both an expression and a statement
            19 => true || { _ = 1; return },
            20 => false || { _ = 1; return },
            // ... or multiple statements
            21 => true || { _ = 1; return; },
            22 => false || { _ = 1; return; },
            23 => true || { return; true },
            24 => true || { return; true },
            _ => true || break,
            //~^ diverging_sub_expression

        };
    }
}

#[allow(unused)]
fn ignore_todo() {
    let x: u32 = todo!();
    println!("{x}");
}
