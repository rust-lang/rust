//@ known-bug: rust-lang/rust#130104

fn main() {
    let non_secure_function =
        core::mem::transmute::<fn() -> _, extern "cmse-nonsecure-call" fn() -> _>;
}
