// pretty-expanded FIXME #23616

struct font<'a> {
    fontbuf: &'a Vec<u8> ,
}

impl<'a> font<'a> {
    pub fn buf(&self) -> &'a Vec<u8> {
        self.fontbuf
    }
}

fn font(fontbuf: &Vec<u8> ) -> font {
    font {
        fontbuf: fontbuf
    }
}

pub fn main() { }
