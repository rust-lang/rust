//@ run-rustfix
#![allow(unused_mut)]
#![allow(dead_code)]

pub trait Layer {
    fn process(&mut self) -> u32;
}

pub struct State {
    layers: Vec<Box<dyn Layer>>,
}

impl State {
    pub fn process(&mut self) -> u32 {
        self.layers.iter().fold(0, |result, mut layer| result + layer.process())
        //~^ ERROR cannot borrow `**layer` as mutable, as it is behind a `&` reference
    }
}

fn main() {}
