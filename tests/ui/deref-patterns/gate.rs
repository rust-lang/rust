// gate-test-string_deref_patterns
fn main() {
    match String::new() {
        "" | _ => {}
        //~^ mismatched types
    }
}
