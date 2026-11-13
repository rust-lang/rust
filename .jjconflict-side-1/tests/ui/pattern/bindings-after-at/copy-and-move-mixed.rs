// Test that mixing `Copy` and non-`Copy` types in `@` patterns is forbidden.

#[derive(Copy, Clone)]
struct C;

struct NC<A, B>(A, B);

fn main() {
    // this compiles
    let a @ NC(b, c) = NC(C, C);

    let a @ NC(b, c @ NC(d, e)) = NC(C, NC(C, C));
    //~^ ERROR use of partially moved value
}
