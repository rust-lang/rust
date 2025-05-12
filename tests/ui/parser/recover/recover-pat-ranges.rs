fn main() {
    match -1 {
        0..=1 => (),
        0..=(1) => (),
        //~^ error: range pattern bounds cannot have parentheses
        (-12)..=4 => (),
        //~^ error: range pattern bounds cannot have parentheses
        (0)..=(-4) => (),
        //~^ error: range pattern bounds cannot have parentheses
        //~| error: range pattern bounds cannot have parentheses
        ..=1 + 2 => (),
        //~^ error: expected a pattern range bound, found an expression
        (4).. => (),
        //~^ error: range pattern bounds cannot have parentheses
        (-4 + 0).. => (),
        //~^ error: expected a pattern range bound, found an expression
        //~| error: range pattern bounds cannot have parentheses
        (1 + 4)...1 * 2 => (),
        //~^ error: expected a pattern range bound, found an expression
        //~| error: expected a pattern range bound, found an expression
        //~| error: range pattern bounds cannot have parentheses
        //~| warning: `...` range patterns are deprecated
        //~| warning: this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
        0.x()..="y".z() => (),
        //~^ error: expected a pattern range bound, found an expression
        //~| error: expected a pattern range bound, found an expression
    };
}

macro_rules! m {
    ($pat:pat) => {};
    (($s:literal)..($e:literal)) => {};
}

m!((7)..(7));
