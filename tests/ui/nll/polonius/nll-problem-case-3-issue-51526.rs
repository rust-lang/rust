#![crate_type = "lib"]

// This is part of a collection of regression tests related to the NLL problem case 3 that was
// deferred from the implementation of the NLL RFC, and left to be implemented by polonius. They are
// from open issues, e.g. tagged fixed-by-polonius, to ensure that the polonius alpha analysis does
// handle them, as does the datalog implementation.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius legacy
//@ [nll] known-bug: #51526
//@ [polonius] check-pass
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] check-pass
//@ [legacy] compile-flags: -Z polonius=legacy

use std::collections::VecDeque;

fn next(queue: &mut VecDeque<u32>, above: u32) -> Option<&u32> {
    let result = loop {
        {
            let next = queue.front()?;
            if *next > above {
                break next;
            }
        }
        queue.pop_front();
    };

    Some(result)
}
