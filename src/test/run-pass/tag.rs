


// -*- rust -*-
tag colour { red(int, int); green; }

fn f() {
    auto x = red(1, 2);
    auto y = green;
    // FIXME: needs structural equality test working.
    // assert (x != y);

}

fn main() { f(); }