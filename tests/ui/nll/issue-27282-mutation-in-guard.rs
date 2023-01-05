#![feature(if_let_guard)]

fn main() {
    match Some(&4) {
        None => {},
        ref mut foo
            if {
                (|| { let bar = foo; bar.take() })();
                //~^ ERROR cannot move out of `foo` in pattern guard
                false
            } => {},
        Some(ref _s) => println!("Note this arm is bogus; the `Some` became `None` in the guard."),
        _ => println!("Here is some supposedly unreachable code."),
    }

    match Some(&4) {
        None => {},
        ref mut foo
            if let Some(()) = {
                (|| { let bar = foo; bar.take() })();
                //~^ ERROR cannot move out of `foo` in pattern guard
                None
            } => {},
        Some(_) => {},
    }
}
