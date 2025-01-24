use std::io::{self, Read};

pub struct Aaaaaaa;

impl Aaaaaaa {
    pub fn bbbbbbb(self) -> u32 {
        1
    }
    pub fn ccccccc(&self) -> bool {
        true
    }
}

impl Read for Aaaaaaa {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        Ok(out.len())
    }
}

pub struct Ddddddd<T>(T);

impl<T: Read> Ddddddd<T> {
    pub fn eeeeeee(_: T) -> u64 {
        1
    }
    pub fn fffffff(_: bool) -> u64 {
        1
    }
    pub fn ggggggg(self) -> u64 {
        1
    }
    pub fn hhhhhhh() -> Self
    where
        T: Default,
    {
        Ddddddd(T::default())
    }
}
