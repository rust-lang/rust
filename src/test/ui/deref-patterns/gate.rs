// gate-test-deref_patterns
fn main() {
    match String::new() {
        "" | _ => {}
        //~^ mismatched types
    }
}
