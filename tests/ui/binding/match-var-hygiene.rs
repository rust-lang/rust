// run-pass
// shouldn't affect evaluation of $ex.
macro_rules! bad_macro { ($ex:expr) => (
    {match 9 {_x => $ex}}
)}

fn main() {
    match 8 {
        _x => assert_eq!(bad_macro!(_x),8)
    }
}
