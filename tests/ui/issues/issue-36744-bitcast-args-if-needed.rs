// run-pass
// This tests for an ICE (and, if ignored, subsequent LLVM abort) when
// a lifetime-parametric fn is passed into a context whose expected
// type has a differing lifetime parameterization.

struct A<'a> {
    _a: &'a i32,
}

fn call<T>(s: T, functions: &Vec<for <'n> fn(&'n T)>) {
    for function in functions {
        function(&s);
    }
}

fn f(a: &A) { println!("a holds {}", a._a); }

fn main() {
    let a = A { _a: &10 };

    let vec: Vec<for <'u,'v> fn(&'u A<'v>)> = vec![f];
    call(a, &vec);
}
