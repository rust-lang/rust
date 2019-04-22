#![feature(box_syntax)]

fn call_f<F:FnOnce() -> isize>(f: F) -> isize {
    f()
}

fn main() {
    let t: Box<_> = box 3;

    call_f(move|| { *t + 1 });
    call_f(move|| { *t + 1 }); //~ ERROR use of moved value
}
