


// -*- rust -*-
tag colour { red(int, int); green; }

fn f() {
    let x = red(1, 2);
    let y = green;
    assert (x != y);
}

fn main() { f(); }
