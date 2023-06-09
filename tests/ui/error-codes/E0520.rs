#![feature(specialization)]
//~^ WARN the feature `specialization` is incomplete

trait SpaceLlama {
    fn fly(&self);
}

impl<T> SpaceLlama for T {
    default fn fly(&self) {}
}

impl<T: Clone> SpaceLlama for T {
    fn fly(&self) {}
}

impl SpaceLlama for i32 {
    default fn fly(&self) {}
    //~^ ERROR E0520
}

fn main() {
}
