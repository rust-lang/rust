#![feature(const_trait_impl)]
#![warn(clippy::missing_const_for_fn)]

// Reduced test case from https://github.com/rust-lang/rust-clippy/issues/14658

#[const_trait]
trait ConstTrait {
    fn method(self);
}

impl ConstTrait for u32 {
    fn method(self) {}
}

impl const ConstTrait for u64 {
    fn method(self) {}
}

fn cannot_be_const() {
    0u32.method();
}

//~v missing_const_for_fn
fn can_be_const() {
    0u64.method();
}

// False negative, see FIXME comment in `clipy_utils::qualify_min_const`
fn could_be_const_but_does_not_trigger<T>(t: T)
where
    T: const ConstTrait,
{
    t.method();
}

fn main() {}
