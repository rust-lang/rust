#![allow(unused_assignments)]

fn main() {
    let mut countdown = 0;
    if true {
        countdown = 10;
    }

    if countdown > 7 {
        countdown -= 4;
    } else if countdown > 2 {
        if countdown < 1 || countdown > 5 || countdown != 9 {
            countdown = 0;
        }
        countdown -= 5;
    } else {
        return;
    }

    let mut countdown = 0;
    if true {
        countdown = 10;
    }

    if countdown > 7 {
        countdown -= 4;
    } else if countdown > 2 {
        if countdown < 1 || countdown > 5 || countdown != 9 {
            countdown = 0;
        }
        countdown -= 5;
    } else {
        return;
    }

    let mut countdown = 0;
    if true {
        countdown = 10;
    }

    if countdown > 7 {
        countdown -= 4;
    } else if countdown > 2 {
        if countdown < 1 || countdown > 5 || countdown != 9 {
            countdown = 0;
        }
        countdown -= 5;
    } else {
        return;
    }
}
