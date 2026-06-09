// Parenthesised let "expressions" are not allowed in guards

#[cfg(false)]
fn un_cfged() {
    match () {
        () if let 0 = 1 => {}
        () if (let 0 = 1) => {}
        //~^ ERROR expected expression, found `let` statement
        () if (((let 0 = 1))) => {}
        //~^ ERROR expected expression, found `let` statement
    }
}

fn main() {
    match () {
        () if let 0 = 1 => {}
        () if (let 0 = 1) => {}
        //~^ ERROR expected expression, found `let` statement
        () if (((let 0 = 1))) => {}
        //~^ ERROR expected expression, found `let` statement
    }
}
