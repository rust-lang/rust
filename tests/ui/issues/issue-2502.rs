//@ check-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]



struct font<'a> {
    fontbuf: &'a Vec<u8> ,
}

impl<'a> font<'a> {
    pub fn buf(&self) -> &'a Vec<u8> {
        self.fontbuf
    }
}

fn font(fontbuf: &Vec<u8> ) -> font<'_> {
    font {
        fontbuf: fontbuf
    }
}

pub fn main() { }
