//@ known-bug: rust-lang/rust#130104

fn main() {
    let non_secure_function =
        core::mem::transmute::<fn() -> _, extern "C-cmse-nonsecure-call" fn() -> _>;
}
