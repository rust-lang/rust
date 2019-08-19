// run-pass
const ARR: [usize; 5] = [5, 4, 3, 2, 1];
const BLA: usize = ARR[ARR[3]];

fn main() {
    assert_eq!(3, BLA);
}
