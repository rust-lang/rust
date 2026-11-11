//@ compile-flags: --crate-type=lib

// A bunch of tests for syntactic forms involving blocks that were
// previously ambiguous (e.g., 'if true { } *val;' gets parsed as a
// binop)


use std::cell::Cell;

fn test1() { let val = &0; { } *val; }

fn test2() -> isize { let val = &0; { } *val }

#[derive(Copy, Clone)]
struct S { eax: isize }

fn test3() {
    let regs = &Cell::new(S {eax: 0});
    match true { true => { } _ => { } }
    regs.set(S {eax: 1});
}

fn test4() -> bool { let regs = &true; if true { } *regs || false }

fn test5() -> (isize, isize) { { } (0, 1) }

fn test6() -> bool { { } (true || false) && true }

fn test7() -> usize {
    let regs = &0;
    match true { true => { } _ => { } }
    (*regs < 2) as usize
}

fn test8() -> isize {
    let val = &0;
    match true {
        true => { }
        _    => { }
    }
    if *val < 1 {
        0
    } else {
        1
    }
}

fn test9() {
    let regs = &Cell::new(0);
    match true { true => { } _ => { } } regs.set(regs.get() + 1);
}

fn test10() -> isize {
    let regs = vec![0];
    match true { true => { } _ => { } }
    regs[0]
}

fn test11() -> Vec<isize> { if true { } vec![1, 2] }
