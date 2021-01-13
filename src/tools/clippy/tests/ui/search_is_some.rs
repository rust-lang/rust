// aux-build:option_helpers.rs
extern crate option_helpers;
use option_helpers::IteratorFalsePositives;

#[warn(clippy::search_is_some)]
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
}
