//@check-pass

fn main() {}

fn issue16330() {
    let a;
    let b;
    if true {
        a = 1;
        b = 2;
    } else {
        a = 3;
        b = 4;
    }

    let a;
    let mut b = 1;
    let c;
    if true {
        b = 1;
        a = 2;
        c = 3;
    } else {
        b = 6;
        a = 4;
        c = 5;
    }

    let b;
    {
        let a;
        let c;
        if true {
            b = 1;
            a = 2;
            c = 3;
        } else {
            b = 6;
            a = 4;
            c = 5;
        }
    }

    let a;
    let b;
    let c;
    match 1 {
        1 => {
            a = 1;
            b = 2;
            c = 3;
        },
        _ if false => {
            a = 4;
            b = 5;
            c = 6;
        },
        _ => {
            a = 7;
            b = 8;
            c = 9;
        },
    }
}
