#![feature(half_open_range_patterns)]

fn main() {}

#[cfg(FALSE)]
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
    }
}
