#![feature(deref_patterns)]
#![allow(incomplete_features, clippy::eq_op)]
#![warn(clippy::single_match_else)]

fn string() {
    match *"" {
        //~^ single_match
        "" => {},
        _ => {},
    }
}
