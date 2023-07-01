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

fn main() { }
