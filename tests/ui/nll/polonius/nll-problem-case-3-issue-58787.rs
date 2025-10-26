#![crate_type = "lib"]

// This is part of a collection of regression tests related to the NLL problem case 3 that was
// deferred from the implementation of the NLL RFC, and left to be implemented by polonius. They are
// from open issues, e.g. tagged fixed-by-polonius, to ensure that the polonius alpha analysis does
// handle them, as does the datalog implementation.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius legacy
//@ [nll] known-bug: #58787
//@ [polonius] check-pass
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] check-pass
//@ [legacy] compile-flags: -Z polonius=legacy

struct Node {
    rest: List,
}

struct List(Option<Box<Node>>);

fn issue_58787(arg: &mut List) {
    let mut list = arg;

    match list.0 {
        Some(ref mut d) => {
            if true {
                list = &mut d.rest;
            }
        }
        None => (),
    }

    match list.0 {
        Some(ref mut d) => {
            list = &mut d.rest;
        }
        None => (),
    }

    match list {
        List(Some(d)) => {
            if true {
                list = &mut d.rest;
            }
        }
        List(None) => (),
    }

    match list {
        List(Some(d)) => {
            list = &mut d.rest;
        }
        List(None) => (),
    }

    match &mut list.0 {
        Some(d) => {
            if true {
                list = &mut d.rest;
            }
        }
        None => (),
    }

    match &mut list.0 {
        Some(d) => {
            list = &mut d.rest;
        }
        None => (),
    }

    list.0 = None;
}
