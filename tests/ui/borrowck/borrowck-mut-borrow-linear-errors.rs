// Test to ensure we only report an error for the first issued loan that
// conflicts with a new loan, as opposed to every issued loan.  This keeps us
// down to O(n) errors (for n problem lines), instead of O(n^2) errors.

fn main() {
    let mut x = 1;
    let mut addr = vec![];
    loop {
        match 1 {
            1 => { addr.push(&mut x); } //~ ERROR [E0499]
            2 => { addr.push(&mut x); } //~ ERROR [E0499]
            _ => { addr.push(&mut x); } //~ ERROR [E0499]
        }
    }
}
