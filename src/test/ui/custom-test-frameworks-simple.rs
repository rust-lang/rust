// compile-flags: --test
// run-pass

#![feature(custom_test_frameworks)]
#![test_runner(crate::foo_runner)]

#[cfg(test)]
fn foo_runner(ts: &[&dyn Fn(usize)->()]) {
    for (i, t) in ts.iter().enumerate() {
        t(i);
    }
}

#[test_case]
fn test1(i: usize) {
    println!("Hi #{}", i);
}

#[test_case]
fn test2(i: usize) {
    println!("Hey #{}", i);
}
