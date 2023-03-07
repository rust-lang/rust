// Part of issue #27300.
// The problem here is that ruby-style closures are parsed as blocks whose
// first statement is a closure. See the issue for more details:
// https://github.com/rust-lang/rust/issues/27300

// Note: this test represents what the compiler currently emits. The error
// message will be improved later.

fn main() {
    let p = Some(45).and_then({
        |x| println!("doubling {}", x);
        Some(x * 2)
        //~^ ERROR: cannot find value `x` in this scope
    });
}
