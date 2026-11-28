#![feature(ergonomic_clones)]
#![expect(incomplete_features)]

fn main() {
    let s = String::new();
    let _ = s.use; //~  ERROR the trait bound `String: UseCloned` is not satisfied
}
