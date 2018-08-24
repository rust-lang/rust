// pretty-expanded FIXME #23616

struct CMap<'a> {
    buf: &'a [u8],
}

fn CMap(buf: &[u8]) -> CMap {
    CMap {
        buf: buf
    }
}

pub fn main() { }
