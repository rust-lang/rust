// check-only
// run-rustfix

fn main() {
    match 3 {
        4 => 1,
        3 => {
            2 //~ ERROR mismatched types
        }
        _ => 2
    }
    match 3 { //~ ERROR mismatched types
        4 => 1,
        3 => 2,
        _ => 2
    }
    let _ = ();
}
