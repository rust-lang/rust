

fn main() {
    let i = 0;
    while i < 20 { i += 1; if i == 10 { break; } }
    assert (i == 10);
    do  { i += 1; if i == 20 { break; } } while i < 30
    assert (i == 20);
    for x: int in [1, 2, 3, 4, 5, 6] { if x == 3 { break; } assert (x <= 3); }
    i = 0;
    while i < 10 { i += 1; if i % 2 == 0 { cont; } assert (i % 2 != 0); }
    i = 0;
    do  { i += 1; if i % 2 == 0 { cont; } assert (i % 2 != 0); } while i < 10
    for x: int in [1, 2, 3, 4, 5, 6] {
        if x % 2 == 0 { cont; }
        assert (x % 2 != 0);
    }
}
