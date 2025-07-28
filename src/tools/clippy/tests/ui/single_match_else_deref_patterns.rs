#![feature(deref_patterns)]
#![allow(
    incomplete_features,
    clippy::eq_op,
    clippy::op_ref,
    clippy::deref_addrof,
    clippy::borrow_deref_ref,
    clippy::needless_if
)]
#![deny(clippy::single_match_else)]

fn string() {
    match *"" {
        //~^ single_match
        "" => {},
        _ => {},
    }

    match *&*&*&*"" {
        //~^ single_match
        "" => {},
        _ => {},
    }

    match ***&&"" {
        //~^ single_match
        "" => {},
        _ => {},
    }

    match *&*&*"" {
        //~^ single_match
        "" => {},
        _ => {},
    }

    match **&&*"" {
        //~^ single_match
        "" => {},
        _ => {},
    }
}

fn int() {
    match &&&1 {
        &&&2 => unreachable!(),
        _ => {
            // ok
        },
    }
    //~^^^^^^ single_match_else
    match &&&1 {
        &&2 => unreachable!(),
        _ => {
            // ok
        },
    }
    //~^^^^^^ single_match_else
    match &&1 {
        &&2 => unreachable!(),
        _ => {
            // ok
        },
    }
    //~^^^^^^ single_match_else
    match &&&1 {
        &2 => unreachable!(),
        _ => {
            // ok
        },
    }
    //~^^^^^^ single_match_else
    match &&1 {
        &2 => unreachable!(),
        _ => {
            // ok
        },
    }
    //~^^^^^^ single_match_else
    match &&&1 {
        2 => unreachable!(),
        _ => {
            // ok
        },
    }
    //~^^^^^^ single_match_else
    match &&1 {
        2 => unreachable!(),
        _ => {
            // ok
        },
    }
    //~^^^^^^ single_match_else
}
