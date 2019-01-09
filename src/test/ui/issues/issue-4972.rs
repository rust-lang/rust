#![feature(box_patterns)]
#![feature(box_syntax)]

trait MyTrait {
    fn dummy(&self) {}
}

pub enum TraitWrapper {
    A(Box<MyTrait+'static>),
}

fn get_tw_map(tw: &TraitWrapper) -> &MyTrait {
    match *tw {
        TraitWrapper::A(box ref map) => map, //~ ERROR cannot be dereferenced
    }
}

pub fn main() {}
