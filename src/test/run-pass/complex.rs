


// -*- rust -*-
type t = int;

fn nothing() { }

fn putstr(str s) { }

fn putint(int i) {
    let int i = 33;
    while (i < 36) { putstr("hi"); i = i + 1; }
}

fn zerg(int i) -> int { ret i; }

fn foo(int x) -> int {
    let t y = x + 2;
    putstr("hello");
    while (y < 10) { putint(y); if (y * 3 == 4) { y = y + 2; nothing(); } }
    let t z;
    z = 0x55;
    foo(z);
    ret 0;
}

fn main() { let int x = 2 + 2; log x; log "hello, world"; log 10; }