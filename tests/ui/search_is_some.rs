// aux-build:option_helpers.rs
#![warn(clippy::search_is_some)]
#![allow(dead_code)]
extern crate option_helpers;
use option_helpers::IteratorFalsePositives;

#[rustfmt::skip]
fn main() {
    let v = vec![3, 2, 1, 0, -1, -2, -3];
    let y = &&42;


    // Check `find().is_some()`, multi-line case.
    let _ = v.iter().find(|&x| {
                              *x < 0
                          }
                   ).is_some();

    // Check `position().is_some()`, multi-line case.
    let _ = v.iter().position(|&x| {
                                  x < 0
                              }
                   ).is_some();

    // Check `rposition().is_some()`, multi-line case.
    let _ = v.iter().rposition(|&x| {
                                   x < 0
                               }
                   ).is_some();

    // Check that we don't lint if the caller is not an `Iterator` or string
    let falsepos = IteratorFalsePositives { foo: 0 };
    let _ = falsepos.find().is_some();
    let _ = falsepos.position().is_some();
    let _ = falsepos.rposition().is_some();
    // check that we don't lint if `find()` is called with
    // `Pattern` that is not a string
    let _ = "hello world".find(|c: char| c == 'o' || c == 'l').is_some();

    // Check `find().is_some()`, single-line case.
    let _ = (0..1).find(|x| **y == *x).is_some(); // one dereference less
    let _ = (0..1).find(|x| *x == 0).is_some();
    let _ = v.iter().find(|x| **x == 0).is_some();
}

#[rustfmt::skip]
fn is_none() {
    let v = vec![3, 2, 1, 0, -1, -2, -3];
    let y = &&42;


    // Check `find().is_none()`, multi-line case.
    let _ = v.iter().find(|&x| {
                              *x < 0
                          }
                   ).is_none();

    // Check `position().is_none()`, multi-line case.
    let _ = v.iter().position(|&x| {
                                  x < 0
                              }
                   ).is_none();

    // Check `rposition().is_none()`, multi-line case.
    let _ = v.iter().rposition(|&x| {
                                   x < 0
                               }
                   ).is_none();

    // Check that we don't lint if the caller is not an `Iterator` or string
    let falsepos = IteratorFalsePositives { foo: 0 };
    let _ = falsepos.find().is_none();
    let _ = falsepos.position().is_none();
    let _ = falsepos.rposition().is_none();
    // check that we don't lint if `find()` is called with
    // `Pattern` that is not a string
    let _ = "hello world".find(|c: char| c == 'o' || c == 'l').is_none();

    // Check `find().is_none()`, single-line case.
    let _ = (0..1).find(|x| **y == *x).is_none(); // one dereference less
    let _ = (0..1).find(|x| *x == 0).is_none();
    let _ = v.iter().find(|x| **x == 0).is_none();
}

#[allow(clippy::clone_on_copy, clippy::map_clone)]
mod issue7392 {
    struct Player {
        hand: Vec<usize>,
    }
    fn filter() {
        let p = Player {
            hand: vec![1, 2, 3, 4, 5],
        };
        let filter_hand = vec![5];
        let _ = p
            .hand
            .iter()
            .filter(|c| filter_hand.iter().find(|cc| c == cc).is_none())
            .map(|c| c.clone())
            .collect::<Vec<_>>();
    }

    struct PlayerTuple {
        hand: Vec<(usize, char)>,
    }
    fn filter_tuple() {
        let p = PlayerTuple {
            hand: vec![(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e')],
        };
        let filter_hand = vec![5];
        let _ = p
            .hand
            .iter()
            .filter(|(c, _)| filter_hand.iter().find(|cc| c == *cc).is_none())
            .map(|c| c.clone())
            .collect::<Vec<_>>();
    }
}
