#![feature(if_let_guard)]
#![allow(unused_mut)]

// Here is arielb1's basic example from rust-lang/rust#27282
// that AST borrowck is flummoxed by:

fn should_reject_destructive_mutate_in_guard() {
    match Some(&4) {
        None => {},
        ref mut foo if {
            (|| { let mut bar = foo; bar.take() })();
            //~^ ERROR cannot move out of `foo` in pattern guard [E0507]
            false } => { },
        Some(s) => std::process::exit(*s),
    }

    match Some(&4) {
        None => {},
        ref mut foo if let Some(()) = {
            (|| { let mut bar = foo; bar.take() })();
            //~^ ERROR cannot move out of `foo` in pattern guard [E0507]
            None } => { },
        Some(s) => std::process::exit(*s),
    }
}

// Here below is a case that needs to keep working: we only use the
// binding via immutable-borrow in the guard, and we mutate in the arm
// body.
fn allow_mutate_in_arm_body() {
    match Some(&4) {
        None => {},
        ref mut foo if foo.is_some() => { foo.take(); () }
        Some(s) => std::process::exit(*s),
    }

    match Some(&4) {
        None => {},
        ref mut foo if let Some(_) = foo => { foo.take(); () }
        Some(s) => std::process::exit(*s),
    }
}

// Here below is a case that needs to keep working: we only use the
// binding via immutable-borrow in the guard, and we move into the arm
// body.
fn allow_move_into_arm_body() {
    match Some(&4) {
        None => {},
        mut foo if foo.is_some() => { foo.unwrap(); () }
        Some(s) => std::process::exit(*s),
    }

    match Some(&4) {
        None => {},
        mut foo if let Some(_) = foo => { foo.unwrap(); () }
        Some(s) => std::process::exit(*s),
    }
}

fn main() {
    should_reject_destructive_mutate_in_guard();
    allow_mutate_in_arm_body();
    allow_move_into_arm_body();
}
