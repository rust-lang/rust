//@ proc-macro: weird-hygiene.rs
//@ ignore-backends: gcc

#![feature(stmt_expr_attributes)]
#![feature(proc_macro_hygiene)]

extern crate weird_hygiene;
use weird_hygiene::*;

macro_rules! other {
    ($tokens:expr) => {
        macro_rules! call_it {
            ($outer_ident:ident) => {
                macro_rules! inner {
                    () => {
                        $outer_ident;
                    }
                }
            }
        }

        #[derive(WeirdDerive)]
        enum MyEnum {
            Value = (stringify!($tokens + hidden_ident), 1).1 //~ ERROR cannot find
        }

        inner!();
    }
}

macro_rules! invoke_it {
    ($token:expr) => {
        #[recollect_attr] {
            $token;
            hidden_ident //~ ERROR cannot find
        }
    }
}

fn main() {
    // `other` and `invoke_it` are both macro_rules! macros,
    // so it should be impossible for them to ever see `hidden_ident`,
    // even if they invoke a proc macro.
    let hidden_ident = "Hello1";
    other!(50);
    invoke_it!(25);
}
