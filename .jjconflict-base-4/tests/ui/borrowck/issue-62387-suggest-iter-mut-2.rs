//@ run-rustfix
#![allow(unused_mut)]
#![allow(dead_code)]
use std::path::PathBuf;

#[derive(Clone)]
struct Container {
    things: Vec<PathBuf>,
}

impl Container {
    fn things(&mut self) -> &[PathBuf] {
        &self.things
    }
}

// contains containers
struct ContainerContainer {
    contained: Vec<Container>,
}

impl ContainerContainer {
    fn contained(&self) -> &[Container] {
        &self.contained
    }

    fn all_the_things(&mut self) -> &[PathBuf] {
        let mut vec = self.contained.clone();
        let _a =
            vec.iter().flat_map(|container| container.things()).cloned().collect::<Vec<PathBuf>>();
        //~^ ERROR cannot borrow `*container` as mutable, as it is behind a `&` reference
        unimplemented!();
    }
}

fn main() {}
