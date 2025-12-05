fn main() {}

#[cfg(false)]
fn syntax() {
    match &0 {
        &0.. | _ => {}
        //~^ ERROR the range pattern here has ambiguous interpretation
        &0..= | _ => {}
        //~^ ERROR the range pattern here has ambiguous interpretation
        //~| ERROR inclusive range with no end
        &0... | _ => {}
        //~^ ERROR inclusive range with no end
    }

    match &0 {
        &..0 | _ => {}
        //~^ ERROR the range pattern here has ambiguous interpretation
        &..=0 | _ => {}
        //~^ ERROR the range pattern here has ambiguous interpretation
        &...0 | _ => {}
        //~^ ERROR the range pattern here has ambiguous interpretation
        //~| ERROR range-to patterns with `...` are not allowed
    }
}
