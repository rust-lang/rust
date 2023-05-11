// compile-flags: -Ztrait-solver=next
// check-pass

fn foo<T>(x: T) {
    std::mem::discriminant(&x);
}

fn main() {}
