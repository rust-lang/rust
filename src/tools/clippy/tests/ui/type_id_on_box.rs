//@run-rustfix

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

fn main() {
    let any_box: Box<dyn Any> = Box::new(0usize);
    let _ = any_box.type_id();
    let _ = TypeId::of::<Box<dyn Any>>(); // Don't lint. We explicitly say "do this instead" if this is intentional
    let _ = (*any_box).type_id();
    let any_box: &Box<dyn Any> = &(Box::new(0usize) as Box<dyn Any>);
    let _ = any_box.type_id(); // 2 derefs are needed here to get to the `dyn Any`

    let b = existential();
    let _ = b.type_id(); // Don't lint.

    let b: SomeBox = Box::new(0usize);
    let _ = b.type_id();

    let b = BadBox(Box::new(0usize));
    let _ = b.type_id(); // Don't lint. This is a call to `<BadBox as Any>::type_id`. Not `std::boxed::Box`!
}
