#![crate_type = "lib"]

// An example from #57165 of the linked-list cursor-like pattern of #46859/#48001, where the
// polonius alpha analysis shows the same imprecision as NLLs, unlike the datalog implementation.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius legacy
//@ [nll] known-bug: #57165
//@ [polonius] known-bug: #57165
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] check-pass
//@ [legacy] compile-flags: -Z polonius=legacy

struct X {
    next: Option<Box<X>>,
}

fn no_control_flow() {
    let mut b = Some(Box::new(X { next: None }));
    let mut p = &mut b;
    while let Some(now) = p {
        p = &mut now.next;
    }
}

// NLLs and polonius alpha fail here
fn conditional() {
    let mut b = Some(Box::new(X { next: None }));
    let mut p = &mut b;
    while let Some(now) = p {
        if true {
            p = &mut now.next;
        }
    }
}

fn conditional_with_indirection() {
    let mut b = Some(Box::new(X { next: None }));
    let mut p = &mut b;
    while let Some(now) = p {
        if true {
            p = &mut p.as_mut().unwrap().next;
        }
    }
}
