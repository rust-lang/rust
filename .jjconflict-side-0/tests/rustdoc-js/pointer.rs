#![feature(extern_types)]

pub fn pinky(input: *const usize, manage: usize) {
    unimplemented!()
}

pub struct Thumb;

impl Thumb {
    pub fn up(this: *const Self, finger: Thumb) {
        unimplemented!()
    }
}

pub enum Index {}

impl Index {
    pub fn point(self, data: *const Index) {
        unimplemented!()
    }
}

pub union Ring {
    magic: u32,
    marriage: f32,
}

impl Ring {
    pub fn wear(this: *mut Self, extra: *const Ring) {
        unimplemented!()
    }
}

extern "C" {
    pub type Middle;
}

pub fn show(left: *const *mut Middle, right: *mut *const Middle) {
    unimplemented!()
}
