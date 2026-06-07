use std::mem::transmute;

enum Never {}

fn foo(x: Never) { //~ERROR: invalid value of type Never
    let ptr = &raw const x;
    println!("{ptr:p}");
}

fn main() {
    let f = unsafe { transmute::<fn(Never), fn(())>(foo) };
    f(());
}
