//@ run-pass

use {
    std::{
        rc::Rc,
        sync::Arc,
    },
};

#[derive(Default)]
struct Ty;

trait Trait {
    fn receive_trait(self: &Arc<Rc<Box<Self>>>) -> u32;
}

const TRAIT_MAGIC: u32 = 42;
const INHERENT_MAGIC: u32 = 1995;

impl Trait for Ty {
    fn receive_trait(self: &Arc<Rc<Box<Self>>>) -> u32 {
        TRAIT_MAGIC
    }
}

impl Ty {
    fn receive_inherent(self: &Arc<Rc<Box<Self>>>) -> u32 {
        INHERENT_MAGIC
    }
}

fn main() {
    let ty = <Arc<Rc<Box<Ty>>>>::default();
    assert_eq!(TRAIT_MAGIC, ty.receive_trait());
    assert_eq!(INHERENT_MAGIC, ty.receive_inherent());
}
