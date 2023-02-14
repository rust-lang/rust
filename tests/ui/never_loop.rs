#![allow(
    clippy::single_match,
    unused_assignments,
    unused_variables,
    clippy::while_immutable_condition
)]

fn test1() {
    let mut x = 0;
    loop {
        // clippy::never_loop
        x += 1;
        if x == 1 {
            return;
        }
        break;
    }
}

fn test2() {
    let mut x = 0;
    loop {
        x += 1;
        if x == 1 {
            break;
        }
    }
}

fn test3() {
    let mut x = 0;
    loop {
        // never loops
        x += 1;
        break;
    }
}

fn test4() {
    let mut x = 1;
    loop {
        x += 1;
        match x {
            5 => return,
            _ => (),
        }
    }
}

fn test5() {
    let i = 0;
    loop {
        // never loops
        while i == 0 {
            // never loops
            break;
        }
        return;
    }
}

fn test6() {
    let mut x = 0;
    'outer: loop {
        x += 1;
        loop {
            // never loops
            if x == 5 {
                break;
            }
            continue 'outer;
        }
        return;
    }
}

fn test7() {
    let mut x = 0;
    loop {
        x += 1;
        match x {
            1 => continue,
            _ => (),
        }
        return;
    }
}

fn test8() {
    let mut x = 0;
    loop {
        x += 1;
        match x {
            5 => return,
            _ => continue,
        }
    }
}

fn test9() {
    let x = Some(1);
    while let Some(y) = x {
        // never loops
        return;
    }
}

fn test10() {
    for x in 0..10 {
        // never loops
        match x {
            1 => break,
            _ => return,
        }
    }
}

fn test11<F: FnMut() -> i32>(mut f: F) {
    loop {
        return match f() {
            1 => continue,
            _ => (),
        };
    }
}

pub fn test12(a: bool, b: bool) {
    'label: loop {
        loop {
            if a {
                continue 'label;
            }
            if b {
                break;
            }
        }
        break;
    }
}

pub fn test13() {
    let mut a = true;
    loop {
        // infinite loop
        while a {
            if true {
                a = false;
                continue;
            }
            return;
        }
    }
}

pub fn test14() {
    let mut a = true;
    'outer: while a {
        // never loops
        while a {
            if a {
                a = false;
                continue;
            }
        }
        break 'outer;
    }
}

// Issue #1991: the outer loop should not warn.
pub fn test15() {
    'label: loop {
        while false {
            break 'label;
        }
    }
}

// Issue #4058: `continue` in `break` expression
pub fn test16() {
    let mut n = 1;
    loop {
        break if n != 5 {
            n += 1;
            continue;
        };
    }
}

// Issue #9001: `continue` in struct expression fields
pub fn test17() {
    struct Foo {
        f: (),
    }

    let mut n = 0;
    let _ = loop {
        break Foo {
            f: if n < 5 {
                n += 1;
                continue;
            },
        };
    };
}

// Issue #9356: `continue` in else branch of let..else
pub fn test18() {
    let x = Some(0);
    let y = 0;
    // might loop
    let _ = loop {
        let Some(x) = x else {
            if y > 0 {
                continue;
            } else {
                return;
            }
        };

        break x;
    };
    // never loops
    let _ = loop {
        let Some(x) = x else {
            return;
        };

        break x;
    };
}

// Issue #9831: unconditional break to internal labeled block
pub fn test19() {
    fn thing(iter: impl Iterator) {
        for _ in iter {
            'b: {
                break 'b;
            }
        }
    }
}

pub fn test20() {
    'a: loop {
        'b: {
            break 'b 'c: {
                break 'a;
            };
        }
    }
}

pub fn test21() {
    loop {
        'a: {
            { }
            break 'a;
        }
    }
}

fn main() {
    test1();
    test2();
    test3();
    test4();
    test5();
    test6();
    test7();
    test8();
    test9();
    test10();
    test11(|| 0);
    test12(true, false);
    test13();
    test14();
}
