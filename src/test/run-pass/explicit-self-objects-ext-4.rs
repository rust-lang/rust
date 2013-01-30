pub trait Reader {
    // FIXME (#2004): Seekable really should be orthogonal.

    /// Read up to len bytes (or EOF) and put them into bytes (which
    /// must be at least len bytes long). Return number of bytes read.
    // FIXME (#2982): This should probably return an error.
    fn read(bytes: &[mut u8], len: uint) -> uint;
}

pub trait ReaderUtil {

    /// Read len bytes into a new vec.
    fn read_bytes(len: uint);
}

impl<T: Reader> T : ReaderUtil {

    fn read_bytes(len: uint) {
        let mut count = self.read(&[0], len);
    }

}

struct S {
    x: int,
    y: int
}

impl S: Reader {
    fn read(bytes: &[mut u8], len: uint) -> uint {
        0
    }
}

fn main() {
    let x = S { x: 1, y: 2 };
    let x = x as @Reader;
    x.read_bytes(0);
}
