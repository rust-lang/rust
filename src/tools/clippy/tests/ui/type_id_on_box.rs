#![warn(clippy::type_id_on_box)]

use std::any::{Any, TypeId};
use std::ops::Deref;

type SomeBox = Box<dyn Any>;

struct BadBox(Box<dyn Any>);

impl Deref for BadBox {
    type Target = Box<dyn Any>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn existential() -> impl Any {
    Box::new(1) as Box<dyn Any>
}

trait AnySubTrait: Any {}
impl<T: Any> AnySubTrait for T {}

fn main() {
    // Don't lint, calling `.type_id()` on a `&dyn Any` does the expected thing
    let ref_dyn: &dyn Any = &42;
    let _ = ref_dyn.type_id();

    let any_box: Box<dyn Any> = Box::new(0usize);
    let _ = any_box.type_id();
    //~^ ERROR: calling `.type_id()` on

    // Don't lint. We explicitly say "do this instead" if this is intentional
    let _ = TypeId::of::<Box<dyn Any>>();
    let _ = (*any_box).type_id();

    // 2 derefs are needed here to get to the `dyn Any`
    let any_box: &Box<dyn Any> = &(Box::new(0usize) as Box<dyn Any>);
    let _ = any_box.type_id();
    //~^ ERROR: calling `.type_id()` on

    let b = existential();
    let _ = b.type_id(); // Don't

    let b: Box<dyn AnySubTrait> = Box::new(1);
    let _ = b.type_id();
    //~^ ERROR: calling `.type_id()` on

    let b: SomeBox = Box::new(0usize);
    let _ = b.type_id();
    //~^ ERROR: calling `.type_id()` on

    let b = BadBox(Box::new(0usize));
    let _ = b.type_id(); // Don't lint. This is a call to `<BadBox as Any>::type_id`. Not `std::boxed::Box`!
}
