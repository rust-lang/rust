#![allow(unused)]

fn main() {
    let a;
    let n = 1;
    match n {
        1 => a = "one",
        _ => {
            a = "two";
        },
    }

    let b;
    if n == 3 {
        b = "four";
    } else {
        b = "five"
    }

    let c;
    if let Some(n) = Some(5) {
        c = n;
    } else {
        c = -50;
    }

    let d;
    if true {
        let temp = 5;
        d = temp;
    } else {
        d = 15;
    }

    let e;
    if true {
        e = format!("{} {}", a, b);
    } else {
        e = format!("{}", c);
    }

    let f;
    match 1 {
        1 => f = "three",
        _ => return,
    }; // has semi

    let g: usize;
    if true {
        g = 5;
    } else {
        panic!();
    }

    println!("{}", a);
}

async fn in_async() -> &'static str {
    async fn f() -> &'static str {
        "one"
    }

    let a;
    let n = 1;
    match n {
        1 => a = f().await,
        _ => {
            a = "two";
        },
    }

    a
}

const fn in_const() -> &'static str {
    const fn f() -> &'static str {
        "one"
    }

    let a;
    let n = 1;
    match n {
        1 => a = f(),
        _ => {
            a = "two";
        },
    }

    a
}

fn does_not_lint() {
    let z;
    if false {
        z = 1;
    }

    let x;
    let y;
    if true {
        x = 1;
    } else {
        y = 1;
    }

    let mut x;
    if true {
        x = 5;
        x = 10 / x;
    } else {
        x = 2;
    }

    let x;
    let _ = match 1 {
        1 => x = 10,
        _ => x = 20,
    };

    // using tuples would be possible, but not always preferable
    let x;
    let y;
    if true {
        x = 1;
        y = 2;
    } else {
        x = 3;
        y = 4;
    }

    // could match with a smarter heuristic to avoid multiple assignments
    let x;
    if true {
        let mut y = 5;
        y = 6;
        x = y;
    } else {
        x = 2;
    }

    let (x, y);
    if true {
        x = 1;
    } else {
        x = 2;
    }
    y = 3;

    macro_rules! assign {
        ($i:ident) => {
            $i = 1;
        };
    }
    let x;
    assign!(x);

    let x;
    if true {
        assign!(x);
    } else {
        x = 2;
    }

    macro_rules! in_macro {
        () => {
            let x;
            x = 1;

            let x;
            if true {
                x = 1;
            } else {
                x = 2;
            }
        };
    }
    in_macro!();

    println!("{}", x);
}
