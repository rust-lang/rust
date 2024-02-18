//@ run-rustfix

fn test1() {
    let mut chars = "Hello".chars();
    for _c in chars.by_ref() {
        chars.next(); //~ ERROR cannot borrow `chars` as mutable more than once at a time
    }
}

fn test2() {
    let v = vec![1, 2, 3];
    let mut iter = v.iter();
    for _i in iter {
        iter.next(); //~ ERROR borrow of moved value: `iter`
    }
}

fn test3() {
    let v = vec![(), (), ()];
    let mut i = v.iter();
    for () in i.by_ref() {
        i.next(); //~ ERROR cannot borrow `i`
    }
}

fn test4() {
    let v = vec![(), (), ()];
    let mut iter = v.iter();
    for () in iter {
        iter.next(); //~ ERROR borrow of moved value: `iter`
    }
}

fn main() {
    test1();
    test2();
    test3();
    test4();
}
