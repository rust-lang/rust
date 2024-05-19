// Check that we do suggest `(ref a, ref b)` here, since `a` and `b`
// are nested within a pattern
fn main() {
    let x = vec![(String::new(), String::new())];
    let (a, b) = x[0]; //~ ERROR cannot move out of index
}
