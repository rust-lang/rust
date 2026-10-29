// Check that or patterns are lowered correctly in `for` loops.
//@ run-pass

fn main() {
    let v = vec![Ok(2), Err(3), Ok(5)];
    let mut w = Vec::new();
    for &(Ok(i) | Err(i)) in &v {
        w.push(i);
    }
    let mut u = Vec::new();
    for Ok(i) | Err(i) in v {
        u.push(i);
    }
    assert_eq!(w, [2, 3, 5]);
    assert_eq!(u, [2, 3, 5]);
}
