#![feature(plugin)]
#![plugin(clippy)]
#![allow(single_match, unused_assignments, unused_variables)]

fn test1() {
    let mut x = 0;
    loop { // never_loop
        x += 1;
        if x == 1 {
            return
        }
        break;
    }
}

fn test2() {
    let mut x = 0;
    loop {
        x += 1;
        if x == 1 {
            break
        }
    }
}

fn test3() {
    let mut x = 0;
    loop { // never loops
        x += 1;
        break
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
	loop { // never loops
        while i == 0 { // never loops
            break
        }
        return
	}
}

fn test6() {
    let mut x = 0;
    'outer: loop { // never loops
        x += 1;
		loop { // never loops
            if x == 5 { break }
			continue 'outer
		}
		return
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
        return
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
    while let Some(y) = x { // never loops
        return
    }
}

fn test10() {
    for x in 0..10 { // never loops
        match x {
            1 => break,
            _ => return,
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
}

