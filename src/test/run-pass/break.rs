

fn main() {
    let mut i = 0;
    while i < 20 { i += 1; if i == 10 { break; } }
    assert (i == 10);
    loop { i += 1; if i == 20 { break; } }
    assert (i == 20);
    for vec::each(~[1, 2, 3, 4, 5, 6]) |x| {
        if x == 3 { break; } assert (x <= 3);
    }
    i = 0;
    while i < 10 { i += 1; if i % 2 == 0 { loop; } assert (i % 2 != 0); }
    i = 0;
    loop { 
        i += 1; if i % 2 == 0 { loop; } assert (i % 2 != 0); 
        if i >= 10 { break; }
    }
    for vec::each(~[1, 2, 3, 4, 5, 6]) |x| {
        if x % 2 == 0 { loop; }
        assert (x % 2 != 0);
    }
}
