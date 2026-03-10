#![crate_type = "lib"]

// This is part of a collection of regression tests related to the NLL problem case 3 that was
// deferred from the implementation of the NLL RFC, and left to be implemented by polonius. They are
// from open issues, e.g. tagged fixed-by-polonius, to ensure that the polonius alpha analysis does
// handle them, as does the datalog implementation.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius legacy
//@ [nll] known-bug: #54663
//@ [polonius] check-pass
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] check-pass
//@ [legacy] compile-flags: -Z polonius=legacy

use std::rc::Rc;

fn foo(x: &mut u8) -> Option<&u8> {
    if let Some(y) = bar(x) {
        return Some(y);
    }
    bar(x)
}

fn bar(x: &mut u8) -> Option<&u8> {
    Some(x)
}

// Adapted from the compiler's `make_chunk_words_mut_for_change` in `bit_set.rs`.
fn make_chunk_words_mut_for_change<'a>(
    self_words: &'a mut Rc<[u64; 32]>,
    would_change: impl Fn(&[u64; 32]) -> bool,
) -> Option<&'a mut [u64; 32]> {
    if let Some(words) = Rc::get_mut(self_words) {
        return Some(words);
    };

    if !would_change(self_words) {
        return None;
    }

    Some(Rc::make_mut(self_words))
}
