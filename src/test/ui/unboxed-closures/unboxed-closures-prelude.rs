// run-pass
// Tests that the re-exports of `FnOnce` et al from the prelude work.

// pretty-expanded FIXME #23616

fn main() {
    let task: Box<dyn Fn(isize) -> isize> = Box::new(|x| x);
    task(0);

    let mut task: Box<dyn FnMut(isize) -> isize> = Box::new(|x| x);
    task(0);

    call(|x| x, 22);
}

fn call<F:FnOnce(isize) -> isize>(f: F, x: isize) -> isize {
    f(x)
}
