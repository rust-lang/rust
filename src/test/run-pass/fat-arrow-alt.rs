// -*- rust -*-
// xfail-pretty

enum color {
    red,
    green,
    blue
}

fn main() {
    log(error, match red {
        red => { 1 }
        green => { 2 }
        blue => { 3 }
    });
}

