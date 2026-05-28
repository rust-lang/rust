//@ run-rustfix
#![feature(explicit_tail_calls)]
#![expect(incomplete_features)]

fn f() {}

fn g() {
    become (&f)() //~ error: tail calls can only be performed with function definitions or pointers
}

fn h() {
    let table = [f as fn()];
    if let Some(fun) = table.get(0) {
        become fun(); //~ error: tail calls can only be performed with function definitions or pointers
    }
}

fn i() {
    become Box::new(&mut &f)(); //~ error: tail calls can only be performed with function definitions or pointers
}

fn main() {
    g();
    h();
    i();
}
