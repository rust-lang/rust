#![deny(dollar_crate_in_matcher)]

macro_rules! direct {
    ($crate) => {};
    //~^ ERROR usage of `$crate` in matcher
    //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

macro_rules! direct_with_fragment_specifier {
    ($crate:tt) => {};
    //~^ ERROR usage of `$crate` in matcher
    //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

macro_rules! indirect {
    ($dollar:tt $krate:tt) => {
        macro_rules! indirect_inner {
            ($dollar $krate) => {}
        }
    };
}

indirect!($crate);
//~^ ERROR usage of `$crate` in matcher
//~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

macro_rules! indirect_with_fragment_specifier {
    ($dollar:tt $krate:tt) => {
        macro_rules! indirect_with_fragment_specifier_inner {
            ($dollar $krate : tt) => {}
        }
    };
}

indirect_with_fragment_specifier!($crate);
//~^ ERROR usage of `$crate` in matcher
//~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

macro_rules! dollar_crate_metavariable {
    ($dol:tt) => {
        macro_rules! dollar_crate_metavariable_inner {
            ($dol $crate) => {}
            //~^ ERROR missing fragment specifier
            //~| ERROR usage of `$crate` in matcher
            //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
        }
    }
}

dollar_crate_metavariable!($);

macro_rules! dollar_crate_metavariable_with_fragment_specifier {
    ($dol:tt) => {
        macro_rules! dollar_crate_metavariable_with_fragment_specifier_inner {
            ($dol $crate : tt) => {}
            //~^ ERROR usage of `$crate` in matcher
            //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
        }
    }
}

dollar_crate_metavariable_with_fragment_specifier!($);

macro_rules! escaped {
    ($$crate) => {};
    //~^ ERROR unexpected token: $
}

escaped!($crate);

fn main() {}
