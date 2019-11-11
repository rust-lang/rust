// check-pass

// Test `Copy` bindings in the rhs of `@` patterns.

#![feature(slice_patterns)]
#![feature(bindings_after_at)]
//~^ WARN the feature `bindings_after_at` is incomplete and may cause the compiler to crash

#[derive(Copy, Clone)]
struct C;

#[derive(Copy, Clone)]
struct P<A, B>(A, B);

enum E<A, B> { L(A), R(B) }

fn main() {
    let a @ b @ c @ d = C;
    let a @ (b, c) = (C, C);
    let a @ P(b, P(c, d)) = P(C, P(C, C));
    let a @ [b, c] = [C, C];
    let a @ [b, .., c] = [C, C, C];
    let a @ &(b, c) = &(C, C);
    let a @ &(b, &P(c, d)) = &(C, &P(C, C));

    use self::E::*;
    match L(C) {
        L(a) | R(a) => {
            let a: C = a;
            drop(a);
            drop(a);
        }
    }
    match R(&L(&C)) {
        L(L(&a)) | L(R(&a)) | R(L(&a)) | R(R(&a)) => {
            let a: C = a;
            drop(a);
            drop(a);
        }
    }
}
