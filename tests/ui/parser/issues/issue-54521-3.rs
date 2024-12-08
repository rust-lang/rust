//@ run-rustfix

// This test checks that the following error is emitted and the suggestion works:
//
// ```
// let _ = vec![1, 2, 3].into_iter().collect::<Vec<usize>>>>();
//                                                        ^^ help: remove extra angle brackets
// ```

fn main() {
    let _ = vec![1, 2, 3].into_iter().collect::<Vec<usize>>>>>>();
    //~^ ERROR unmatched angle bracket

    let _ = vec![1, 2, 3].into_iter().collect::<Vec<usize>>>>>();
    //~^ ERROR unmatched angle bracket

    let _ = vec![1, 2, 3].into_iter().collect::<Vec<usize>>>>();
    //~^ ERROR unmatched angle bracket

    let _ = vec![1, 2, 3].into_iter().collect::<Vec<usize>>>();
    //~^ ERROR unmatched angle bracket
}
