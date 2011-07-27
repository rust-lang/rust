


// -*- rust -*-
tag colour { red(int, int); green; }

fn f() {
    let x = red(1, 2);
    let y = green;
    // FIXME: needs structural equality test working.
    // assert (x != y);

}

fn main() { f(); }