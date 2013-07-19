#[no_mangle]
fn test(x: int, y: int) -> int {
    match x {
        1 => y,
        2 => y*2,
        4 => y*3,
        _ => 11
    }
}
