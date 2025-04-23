//@ edition: 2024

fn let_or_guard(x: Result<Option<i32>, ()>) {
    match x {
        Ok(opt) if let Some(4) = opt || false  => {}
        //~^ ERROR expected expression, found `let` statement
        _ => {}
    }
}

fn hiding_unsafe_mod(x: Result<Option<i32>, ()>) {
    match x {
        Ok(opt)
            if {
                unsafe mod a {};
                //~^ ERROR module cannot be declared unsafe
                false
            } => {}
        _ => {}
    }
}

fn main() {}
