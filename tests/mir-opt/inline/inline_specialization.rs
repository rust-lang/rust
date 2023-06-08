// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
#![feature(specialization)]

// EMIT_MIR inline_specialization.main.Inline.diff
fn main() {
    let x = <Vec::<()> as Foo>::bar();
}

trait Foo {
    fn bar() -> u32;
}

impl<T> Foo for Vec<T> {
    #[inline(always)]
    default fn bar() -> u32 { 123 }
}
