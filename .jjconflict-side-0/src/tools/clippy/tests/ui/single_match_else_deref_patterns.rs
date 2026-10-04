#![feature(deref_patterns)]
#![warn(clippy::single_match_else)]
#![allow(clippy::eq_op, clippy::needless_ifs, clippy::op_ref)]
#![expect(clippy::borrow_deref_ref, clippy::deref_addrof)]

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
