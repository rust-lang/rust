#[no_mangle]
fn test(x: &[int]) -> int {
    let mut y = 0;
    let mut i = 0;
    while (i < x.len()) {
        y += x[i];
        i += 1;
    }
    y
}
