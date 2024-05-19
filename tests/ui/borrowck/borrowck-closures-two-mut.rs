// Tests that two closures cannot simultaneously have mutable
// access to the variable, whether that mutable access be used
// for direct assignment or for taking mutable ref. Issue #6801.



fn to_fn_mut<F: FnMut()>(f: F) -> F { f }

fn a() {
    let mut x = 3;
    let c1 = to_fn_mut(|| x = 4);
    let c2 = to_fn_mut(|| x = 5); //~ ERROR cannot borrow `x` as mutable more than once
    drop((c1, c2));
}

fn set(x: &mut isize) {
    *x = 4;
}

fn b() {
    let mut x = 3;
    let c1 = to_fn_mut(|| set(&mut x));
    let c2 = to_fn_mut(|| set(&mut x)); //~ ERROR cannot borrow `x` as mutable more than once
    drop((c1, c2));
}

fn c() {
    let mut x = 3;
    let c1 = to_fn_mut(|| x = 5);
    let c2 = to_fn_mut(|| set(&mut x)); //~ ERROR cannot borrow `x` as mutable more than once
    drop((c1, c2));
}

fn d() {
    let mut x = 3;
    let c1 = to_fn_mut(|| x = 5);
    let c2 = to_fn_mut(|| { let _y = to_fn_mut(|| set(&mut x)); }); // (nested closure)
    //~^ ERROR cannot borrow `x` as mutable more than once
    drop((c1, c2));
}

fn g() {
    struct Foo {
        f: Box<isize>
    }

    let mut x: Box<_> = Box::new(Foo { f: Box::new(3) });
    let c1 = to_fn_mut(|| set(&mut *x.f));
    let c2 = to_fn_mut(|| set(&mut *x.f));
    //~^ ERROR cannot borrow `x` as mutable more than once
    drop((c1, c2));
}

fn main() {
}
