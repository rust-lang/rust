

fn main() {
    let x = ~[1, 2, 3];
    let y = 0;
    for i: int  in x { log i; y += i; }
    log y;
    assert (y == 6);
    let s = "hello there";
    let i: int = 0;
    for c: u8  in s {
        if i == 0 { assert (c == 'h' as u8); }
        if i == 1 { assert (c == 'e' as u8); }
        if i == 2 { assert (c == 'l' as u8); }
        if i == 3 { assert (c == 'l' as u8); }
        if i == 4 { assert (c == 'o' as u8); }
        // ...

        i += 1;
        log i;
        log c;
    }
    assert (i == 11);
}