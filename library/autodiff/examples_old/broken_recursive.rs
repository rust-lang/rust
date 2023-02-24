#![feature(bench_black_box)]
use autodiff::autodiff;

// TODO: As seen by the bloated code generated for the iterative version,
// we definetly have to disable unroll, slpvec, loop-vec before AD.
// We also should check if we have other opts that Julia, C++, Fortran etc. don't have
// and which could make our input code more "complex".
// We then however have to start doing whole-module opt after AD to re-include them,
// instead of just using enzyme to optimize the generated function.

#[autodiff(d_power_recursive, Forward, DuplicatedNoNeed)]
fn power_recursive(#[dup] a: f64, n: i32) -> f64 {
    if n == 0 {
        return 1.0;
    }
    return a * power_recursive(a, n - 1);
}

#[autodiff(d_power_iterative, Reverse, DuplicatedNoNeed)]
fn power_iterative(#[active] a: f64, n: i32) -> f64 {
    let mut res = 1.0;
    for _ in 0..n {
        res *= a;
    }
    res
}

fn main() {
    // d/dx x^n = n * x^(n-1)
    let n = 4;
    let nf = n as f64;
    let a = 1.337;
    assert!(power_recursive(a, n) == power_iterative(a, n));
    let dpr = d_power_recursive(a, 1.0, n);
    let dpi = d_power_iterative(a, n, 1.0);
    let control = nf * a.powi(n - 1);
    dbg!(dpr);
    dbg!(dpi);
    dbg!(control);
    assert!(dpr == control);
    assert!(dpi == control);
}

// Again, for the curious. We can find n * x^(n-1) nicely in the LLVM-IR
//
// define internal double @fwddiffe_ZN9recursive15power_recursive17h789de751cfc6154dE(double %0, double %1, i32 %2) unnamed_addr #8 {
// => if (n == 0) goto 5: and return 0. Correct, since for n==0 we have 0 * x ^ (0-1) = 0
// => if (n != 0) goto 7:
//   %4 = icmp eq i32 %2, 0
//   br i1 %4, label %5, label %7
//
// 5:                                                ; preds = %7, %3
//   %6 = phi fast double [ %14, %7 ], [ 0.000000e+00, %3 ]
//   ret double %6
//
// 7:                                                ; preds = %3
// => reduce n by 1,
//   %8 = add i32 %2, -1
//   %9 = call { double, double } @fwddiffe_ZN9recursive15power_recursive17h789de751cfc6154dE.1229(double %0, double %1, i32 %8)
//   %10 = extractvalue { double, double } %9, 0
//   %11 = extractvalue { double, double } %9, 1
//   %12 = fmul fast double %11, %0
//   %13 = fmul fast double %1, %10
//   %14 = fadd fast double %12, %13
//   br label %5
// }
