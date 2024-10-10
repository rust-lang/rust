//@ check-pass

extern "C" {
    fn test();
}

fn a() {}

extern "C" fn c() {}

fn main() {
    type F = fn();
    let f: F = a;
    let g: F = f;

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

    let cfn: extern "C" fn() = c;
    let _ = cfn == c;
    //~^ WARN function pointer comparisons

    let t: unsafe extern "C" fn() = test;
    let _ = t == test;
    //~^ WARN function pointer comparisons
}
