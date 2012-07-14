

fn main() {
    let x = ~[1, 2, 3];
    let mut y = 0;
    for x.each |i| { log(debug, i); y += i; }
    log(debug, y);
    assert (y == 6);
    let s = ~"hello there";
    let mut i: int = 0;
    for str::each(s) |c| {
        if i == 0 { assert (c == 'h' as u8); }
        if i == 1 { assert (c == 'e' as u8); }
        if i == 2 { assert (c == 'l' as u8); }
        if i == 3 { assert (c == 'l' as u8); }
        if i == 4 { assert (c == 'o' as u8); }
        // ...

        i += 1;
        log(debug, i);
        log(debug, c);
    }
    assert (i == 11);
}
