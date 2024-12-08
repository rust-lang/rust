// Ensure macro metavariables are compared with legacy hygiene

#![feature(rustc_attrs)]

//@ run-pass

macro_rules! make_mac {
    ( $($dollar:tt $arg:ident),+ ) => {
        macro_rules! mac {
            ( $($dollar $arg : ident),+ ) => {
                $( $dollar $arg )-+
            }
        }
    }
}

macro_rules! show_hygiene {
    ( $dollar:tt $arg:ident ) => {
        make_mac!($dollar $arg, $dollar arg);
    }
}

show_hygiene!( $arg );

fn main() {
    let x = 5;
    let y = 3;
    assert_eq!(2, mac!(x, y));
}
