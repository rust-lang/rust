#![feature(try_blocks)]
#![allow(
    clippy::eq_op,
    clippy::single_match,
    unused_assignments,
    unused_variables,
    clippy::while_immutable_condition
)]
//@no-rustfix
fn test1() {
    let mut x = 0;
    loop {
        //~^ never_loop

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
        //~^ never_loop

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
        //~^ never_loop

        // never loops
        while i == 0 {
            //~^ never_loop

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
            //~^ never_loop

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
        //~^ never_loop

        // never loops
        return;
    }
}

fn test10() {
    for x in 0..10 {
        //~^ never_loop

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
        //~^ never_loop

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
            //~^ never_loop

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
        //~^ never_loop

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
        //~^ never_loop

        'b: {
            break 'b 'c: {
                break 'a;
                //~^ diverging_sub_expression
            };
        }
    }
}

pub fn test21() {
    loop {
        'a: {
            {}
            break 'a;
        }
    }
}

// Issue 10304: code after break from block was not considered
// unreachable code and was considered for further analysis of
// whether the loop would ever be executed or not.
pub fn test22() {
    for _ in 0..10 {
        'block: {
            break 'block;
            return;
        }
        println!("looped");
    }
}

pub fn test23() {
    for _ in 0..10 {
        'block: {
            for _ in 0..20 {
                //~^ never_loop

                break 'block;
            }
        }
        println!("looped");
    }
}

pub fn test24() {
    'a: for _ in 0..10 {
        'b: {
            let x = Some(1);
            match x {
                None => break 'a,
                Some(_) => break 'b,
            }
        }
    }
}

// Do not lint, we can evaluate `true` to always succeed thus can short-circuit before the `return`
pub fn test25() {
    loop {
        'label: {
            if const { true } {
                break 'label;
            }
            return;
        }
    }
}

pub fn test26() {
    loop {
        'label: {
            if 1 == 1 {
                break 'label;
            }
            return;
        }
    }
}

pub fn test27() {
    loop {
        'label: {
            let x = true;
            if x {
                break 'label;
            }
            return;
        }
    }
}

// issue 11004
pub fn test29() {
    loop {
        'label: {
            if true {
                break 'label;
            }
            return;
        }
    }
}

pub fn test30() {
    'a: loop {
        'b: {
            for j in 0..2 {
                if j == 1 {
                    break 'b;
                }
            }
            break 'a;
        }
    }
}

pub fn test31(b: bool) {
    'a: loop {
        'b: {
            'c: loop {
                //~^ never_loop

                if b { break 'c } else { break 'b }
            }
            continue 'a;
        }
        break 'a;
    }
}

pub fn test32() {
    loop {
        //~^ never_loop

        panic!("oh no");
    }
    loop {
        //~^ never_loop

        unimplemented!("not yet");
    }
    loop {
        // no error
        todo!("maybe later");
    }
}

pub fn issue12205() -> Option<()> {
    loop {
        let _: Option<_> = try {
            None?;
            return Some(());
        };
    }
}

fn stmt_after_return() {
    for v in 0..10 {
        //~^ never_loop
        break;
        println!("{v}");
    }
}

fn loop_label() {
    'outer: for v in 0..10 {
        //~^ never_loop
        loop {
            //~^ never_loop
            break 'outer;
        }
        return;
    }

    for v in 0..10 {
        //~^ never_loop
        'inner: loop {
            //~^ never_loop
            break 'inner;
        }
        return;
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

fn issue15059() {
    'a: for _ in 0..1 {
        //~^ never_loop
        break 'a;
    }

    let mut b = 1;
    'a: for i in 0..1 {
        //~^ never_loop
        match i {
            0 => {
                b *= 2;
                break 'a;
            },
            x => {
                b += x;
                break 'a;
            },
        }
    }

    #[allow(clippy::unused_unit)]
    for v in 0..10 {
        //~^ never_loop
        break;
        println!("{v}");
        // This is comment and should be kept
        println!("This is a comment");
        ()
    }
}
