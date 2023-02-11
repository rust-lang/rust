// run-pass
// shouldn't affect evaluation of $ex:

macro_rules! bad_macro {
    ($ex:expr) => ({let _x = 9; $ex})
}

pub fn main() {
    let _x = 8;
    assert_eq!(bad_macro!(_x),8)
}
