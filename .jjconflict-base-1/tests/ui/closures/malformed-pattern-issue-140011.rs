//@compile-flags: -Wrust-2021-incompatible-closure-captures
enum B {
    C(D), //~ ERROR: cannot find type `D` in this scope
    E(F),
}
struct F;
fn f(h: B) {
    || {
        let B::E(a) = h; //~ ERROR: refutable pattern in local binding
    };
}

fn main() {}
