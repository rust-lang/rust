macro_rules! pat {
    () => { Some(_) }
}

fn main() {
    match Some(false) {
        Some(_)
        //~^ ERROR `match` arm with no body
        //~| HELP add a body after the pattern
    }
    match Some(false) {
        Some(_)
        _ => {}
        //~^ ERROR expected one of
    }
    match Some(false) {
        Some(_),
        //~^ ERROR unexpected `,` in pattern
        //~| HELP try adding parentheses to match on a tuple
        //~| HELP or a vertical bar to match on alternative
    }
    match Some(false) {
        Some(_),
        //~^ ERROR unexpected `,` in pattern
        //~| HELP try adding parentheses to match on a tuple
        //~| HELP or a vertical bar to match on alternative
        _ => {}
    }
    match Some(false) {
        Some(_) if true
        //~^ ERROR `match` arm with no body
        //~| HELP add a body after the pattern
    }
    match Some(false) {
        Some(_) if true
        _ => {}
        //~^ ERROR expected one of
    }
    match Some(false) {
        Some(_) if true,
        //~^ ERROR `match` arm with no body
        //~| HELP add a body after the pattern
    }
    match Some(false) {
        Some(_) if true,
        //~^ ERROR `match` arm with no body
        //~| HELP add a body after the pattern
        _ => {}
    }
    match Some(false) {
        pat!()
        //~^ ERROR `match` arm with no body
        //~| HELP add a body after the pattern
    }
    match Some(false) {
        pat!(),
        //~^ ERROR `match` arm with no body
        //~| HELP add a body after the pattern
    }
    match Some(false) {
        pat!() if true,
        //~^ ERROR `match` arm with no body
        //~| HELP add a body after the pattern
    }
    match Some(false) {
        pat!()
        //~^ ERROR expected `,` following `match` arm
        //~| HELP missing a comma here
        _ => {}
    }
    match Some(false) {
        pat!(),
        //~^ ERROR `match` arm with no body
        //~| HELP add a body after the pattern
        _ => {}
    }
}
