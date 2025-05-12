//@ run-pass

fn or_at(x: Result<u32, u32>) -> u32 {
    match x {
        Ok(x @ 4) | Err(x @ (6 | 8)) => x,
        Ok(x @ 1 | x @ 2) => x,
        Err(x @ (0..=10 | 30..=40)) if x % 2 == 0 => x + 100,
        Err(x @ 0..=40) => x + 200,
        _ => 500,
    }
}

fn main() {
    assert_eq!(or_at(Ok(1)), 1);
    assert_eq!(or_at(Ok(2)), 2);
    assert_eq!(or_at(Ok(3)), 500);
    assert_eq!(or_at(Ok(4)), 4);
    assert_eq!(or_at(Ok(5)), 500);
    assert_eq!(or_at(Ok(6)), 500);
    assert_eq!(or_at(Err(1)), 201);
    assert_eq!(or_at(Err(2)), 102);
    assert_eq!(or_at(Err(3)), 203);
    assert_eq!(or_at(Err(4)), 104);
    assert_eq!(or_at(Err(5)), 205);
    assert_eq!(or_at(Err(6)), 6);
    assert_eq!(or_at(Err(7)), 207);
    assert_eq!(or_at(Err(8)), 8);
    assert_eq!(or_at(Err(20)), 220);
    assert_eq!(or_at(Err(34)), 134);
    assert_eq!(or_at(Err(50)), 500);
}
