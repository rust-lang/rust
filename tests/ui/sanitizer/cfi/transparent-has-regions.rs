//@ needs-sanitizer-cfi
//@ compile-flags: -Ccodegen-units=1 -Clto -Ctarget-feature=-crt-static -Zsanitizer=cfi
//@ no-prefer-dynamic
//@ only-x86_64-unknown-linux-gnu
//@ build-pass

pub trait Trait {}

impl Trait for i32 {}

#[repr(transparent)]
struct BoxedTrait(Box<dyn Trait + 'static>);

fn hello(x: BoxedTrait) {}

fn main() {
    hello(BoxedTrait(Box::new(1)));
}
