#![feature(fn_traits)]

// Ensure that invoking a closure counts as a unique immutable borrow

type Fn<'a> = Box<dyn FnMut() + 'a>;

struct Test<'a> {
    f: Box<dyn FnMut() + 'a>
}

fn call<F>(mut f: F) where F: FnMut(Fn) {
    f(Box::new(|| {
    //~^ ERROR: cannot borrow `f` as mutable more than once
        f((Box::new(|| {})))
    }));
}

fn test1() {
    call(|mut a| {
        a.call_mut(());
    });
}

fn test2<F>(f: &F) where F: FnMut() {
    (*f)();
    //~^ ERROR cannot borrow `*f` as mutable, as it is behind a `&` reference
}

fn test3<F>(f: &mut F) where F: FnMut() {
    (*f)();
}

fn test4(f: &Test) {
    f.f.call_mut(())
    //~^ ERROR: cannot borrow `f.f` as mutable, as it is behind a `&` reference
}

fn test5(f: &mut Test) {
    f.f.call_mut(())
}

fn test6() {
    let mut f = || {};
    (|| {
        f();
    })();
}

fn test7() {
    fn foo<F>(_: F) where F: FnMut(Box<dyn FnMut(isize)>, isize) {}
    let s = String::new();  // Capture to make f !Copy
    let mut f = move |g: Box<dyn FnMut(isize)>, b: isize| {
        let _ = s.len();
    };
    f(Box::new(|a| {
        //~^ ERROR cannot move out of `f` because it is borrowed
        foo(f);
        //~^ ERROR cannot move out of `f`, a captured variable in an `FnMut` closure
    }), 3);
}

fn main() {}
