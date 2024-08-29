#![feature(extern_types)]

pub fn pinky(input: &usize, manage: usize) {
    unimplemented!()
}

pub struct Thumb;

impl Thumb {
    pub fn up(&self, finger: Thumb) {
        unimplemented!()
    }
}

pub enum Index {}

impl Index {
    pub fn point(self, data: &Index) {
        unimplemented!()
    }
}

pub union Ring {
    magic: u32,
    marriage: f32,
}

impl Ring {
    pub fn wear(&mut self, extra: &Ring) {
        unimplemented!()
    }
}

extern "C" {
    pub type Middle;
}

pub fn show(left: &&mut Middle, right: &mut &Middle) {
    unimplemented!()
}
