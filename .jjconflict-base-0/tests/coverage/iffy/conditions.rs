#![allow(unused_assignments, unused_variables)]

fn main() {
    let mut countdown = 0;
    if true {
        countdown = 10;
    }

    const B: u32 = 100;
    let x = if countdown > 7 {
        countdown -= 4;
        B
    } else if countdown > 2 {
        if countdown < 1 || countdown > 5 || countdown != 9 {
            countdown = 0;
        }
        countdown -= 5;
        countdown
    } else {
        return;
    };

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

    if true {
        let mut countdown = 0;
        if true {
            countdown = 10;
        }

        if countdown > 7 {
            countdown -= 4;
        }
        //
        else if countdown > 2 {
            if countdown < 1 || countdown > 5 || countdown != 9 {
                countdown = 0;
            }
            countdown -= 5;
        } else {
            return;
        }
    }

    let mut countdown = 0;
    if true {
        countdown = 1;
    }

    let z = if countdown > 7 {
        countdown -= 4;
    } else if countdown > 2 {
        if countdown < 1 || countdown > 5 || countdown != 9 {
            countdown = 0;
        }
        countdown -= 5;
    } else {
        let should_be_reachable = countdown;
        println!("reached");
        return;
    };

    let w = if countdown > 7 {
        countdown -= 4;
    } else if countdown > 2 {
        if countdown < 1 || countdown > 5 || countdown != 9 {
            countdown = 0;
        }
        countdown -= 5;
    } else {
        return;
    };
}
