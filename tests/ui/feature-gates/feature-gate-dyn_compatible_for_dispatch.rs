// Test that the use of the dyn-incompatible trait objects
// are gated by the `dyn_compatible_for_dispatch` feature gate.

trait DynIncompatible1: Sized {}

trait DynIncompatible2 {
    fn static_fn() {}
}

trait DynIncompatible3 {
    fn foo<T>(&self);
}

trait DynIncompatible4 {
    fn foo(&self, s: &Self);
}

fn takes_dyn_incompatible_ref<T>(obj: &dyn DynIncompatible1) {
    //~^ ERROR E0038
}

fn return_dyn_incompatible_ref() -> &'static dyn DynIncompatible2 {
    //~^ ERROR E0038
    loop {}
}

fn takes_dyn_incompatible_box(obj: Box<dyn DynIncompatible3>) {
    //~^ ERROR E0038
}

fn return_dyn_incompatible_rc() -> std::rc::Rc<dyn DynIncompatible4> {
    //~^ ERROR E0038
    loop {}
}

trait Trait {}

impl Trait for dyn DynIncompatible1 {}
//~^ ERROR E0038

fn main() {}
