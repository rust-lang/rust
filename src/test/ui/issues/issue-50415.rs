// run-pass
fn main() {
    // -------- Simplified test case --------

    let _ = || 0..=1;

    // -------- Original test case --------

    let full_length = 1024;
    let range = {
        // do some stuff, omit here
        None
    };

    let range = range.map(|(s, t)| s..=t).unwrap_or(0..=(full_length-1));

    assert_eq!(range, 0..=1023);
}
