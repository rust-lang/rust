//@ run-pass
// shouldn't affect evaluation of $ex:
macro_rules! bad_macro {
    ($ex:expr) => ({(|_x| { $ex }) (9) })
}

fn takes_x(_x : isize) {
    assert_eq!(bad_macro!(_x),8);
}
fn main() {
    takes_x(8);
}
