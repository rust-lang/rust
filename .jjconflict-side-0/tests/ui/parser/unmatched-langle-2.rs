// When there are too many opening `<`s, the compiler would previously
// suggest nonsense if the `<`s were interspersed with other tokens:
//
//   error: unmatched angle brackets
//    --> unmatched-langle.rs:2:10
//     |
//   2 |     foo::<Ty<<<i32>();
//     |          ^^^ help: remove extra angle brackets
//
// This test makes sure that this is no longer happening.

fn main() {
    foo::<Ty<<<i32>();
    //~^ ERROR: expected `::`, found `(`
}
