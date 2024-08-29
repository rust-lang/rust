//@ known-bug: rust-lang/rust#125655

fn main() {
    static foo: dyn Fn() -> u32 = || -> u32 {
        ...
        0
    };
}
