//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

struct Test {
    func: Box<dyn FnMut() + 'static>,
}

fn main() {
    let closure: Box<dyn Fn() + 'static> = Box::new(|| ());
    let test = Box::new(Test { func: closure });
}
