//@ run-rustfix

#![allow(unused)]

fn main() {
    let n = 1;
    let m = 2;
    let x = {
        if n == 0 {
            break 1; //~ ERROR [E0268]
        } else {
            break 2;
        }
    };

    let y = {
        if n == 0 {
            break 1; //~ ERROR [E0268]
        }
        break 0;
    };

    let z = {
        if n == 0 {
            if m > 1 {
                break 3; //~ ERROR [E0268]
            }
        }
        break 1;
    };
}
