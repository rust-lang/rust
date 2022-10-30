#![feature(box_patterns)]

trait MyTrait {
    fn dummy(&self) {}
}

pub enum TraitWrapper {
    A(Box<dyn MyTrait + 'static>),
}

fn get_tw_map(tw: &TraitWrapper) -> &dyn MyTrait {
    match *tw {
        TraitWrapper::A(box ref map) => map, //~ ERROR cannot be dereferenced
    }
}

pub fn main() {}
