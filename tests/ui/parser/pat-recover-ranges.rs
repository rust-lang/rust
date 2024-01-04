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
    };
}

macro_rules! m {
    ($pat:pat) => {};
    (($s:literal)..($e:literal)) => {};
}

m!((7)..(7));
