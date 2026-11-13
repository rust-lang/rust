//@ check-pass
//@ run-rustfix

extern "C" {
    fn test();
}

fn a() {}

extern "C" fn c() {}

extern "C" fn args(_a: i32) -> i32 { 0 }

struct A {
    f: fn(),
}

fn main() {
    let f: fn() = a;
    let g: fn() = f;

    let a1 = A { f };
    let a2 = A { f };

    let _ = f == a;
    //~^ WARN function pointer comparisons
    let _ = f != a;
    //~^ WARN function pointer comparisons
    let _ = f == g;
    //~^ WARN function pointer comparisons
    let _ = f == f;
    //~^ WARN function pointer comparisons
    let _ = g == g;
    //~^ WARN function pointer comparisons
    let _ = g == g;
    //~^ WARN function pointer comparisons
    let _ = &g == &g;
    //~^ WARN function pointer comparisons
    let _ = a as fn() == g;
    //~^ WARN function pointer comparisons

    let cfn: extern "C" fn() = c;
    let _ = cfn == c;
    //~^ WARN function pointer comparisons

    let argsfn: extern "C" fn(i32) -> i32 = args;
    let _ = argsfn == args;
    //~^ WARN function pointer comparisons

    let t: unsafe extern "C" fn() = test;
    let _ = t == test;
    //~^ WARN function pointer comparisons

    let _ = a1.f == a2.f;
    //~^ WARN function pointer comparisons
}
