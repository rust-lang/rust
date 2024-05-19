//@ run-pass
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
pub struct Registry<'a> {
    listener: &'a mut (),
}

pub struct Listener<'a> {
    pub announce: Option<Box<dyn FnMut(&mut Registry) + 'a>>,
    pub remove: Option<Box<dyn FnMut(&mut Registry) + 'a>>,
}

impl<'a> Drop for Registry<'a> {
    fn drop(&mut self) {}
}

fn main() {
    let mut registry_listener = Listener {
        announce: None,
        remove: None,
    };
}
