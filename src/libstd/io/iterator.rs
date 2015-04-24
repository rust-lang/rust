#![crate_type = "lib"]

use prelude::v1::*;
use io::prelude::*;

use cmp;
use io::{self};
use ptr;
use slice;
use sync;

/// An `IterReader` implements std::io::Read and std::io::BufRead for an iterator of byte sequences.
pub struct IterReader<V, I> where V : AsRef<[u8]>, I : Iterator<Item=V> {
    iter: I,        // The underlying source of the bytes we will read.
    buf: Option<V>, // Contains the last item that we got from the iterator.
    pos: usize,     // The number of bytes that have already been read from buf.
    closed: bool,
}

impl <V : AsRef<[u8]>, I : Iterator<Item=V>> IterReader<V, I> {
    pub fn new(iter: I) -> IterReader<V, I> {
        IterReader { iter: iter, buf: None, pos: 0, closed: false, }
    }

    fn buffered(&self) -> &[u8] {
        match self.buf {
          Some(ref buf) => buf.as_ref()[self.pos..].as_ref(),
          None => &[],
        }
    }

    fn needs_refill(&self) -> bool {
        !self.closed && self.buffered().len() == 0
    }

    fn refill(&mut self) {
        self.pos = 0;
        self.buf = self.iter.next();
        self.closed = self.buf.is_none()
    }
}

impl <V : AsRef<[u8]>, I : Iterator<Item=V>> io::BufRead for IterReader<V, I> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        while self.needs_refill() { // We loop in case the last item was empty.
            self.refill();
        }
        Ok(self.buffered())
    }

    fn consume(&mut self, amt: usize) { self.pos += amt; }
}

impl <V : AsRef<[u8]>, I : Iterator<Item=V>> io::Read for IterReader<V, I> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.needs_refill() {
            self.refill();
        }
        let amt = cmp::min(self.buffered().len(), buf.len());
        slice::bytes::copy_memory(&self.buffered()[..amt], &mut buf[..amt]);
        self.consume(amt);
        Ok(amt)
    }
}

#[test]
fn test_read_vec() {
    use io::Read;

    let pieces = vec![
        vec![1, 2],
        vec![],
        vec![3, 4, 5],
        vec![6, 7, 8, 9],
        vec![10],
    ];

    let mut reader = IterReader::new(pieces.into_iter());

    let mut buf = [0; 3];

    assert_eq!(0, reader.read(&mut []).unwrap());

    assert_eq!(2, reader.read(&mut buf).unwrap());
    assert_eq!([1,2,0], buf);

    assert_eq!(0, reader.read(&mut buf).unwrap());
    assert_eq!([1,2,0], buf);

    assert_eq!(3, reader.read(&mut buf).unwrap());
    assert_eq!([3,4,5], buf);

    assert_eq!(3, reader.read(&mut buf).unwrap());
    assert_eq!([6,7,8], buf);

    assert_eq!(1, reader.read(&mut buf).unwrap());
    assert_eq!([9,7,8], buf);

    assert_eq!(1, reader.read(&mut buf).unwrap());
    assert_eq!([10,7,8], buf);

    assert_eq!(0, reader.read(&mut buf).unwrap());
    assert_eq!(0, reader.read(&mut buf).unwrap());
}

#[test]
fn test_read_array() {
    use io::Read;

    let pieces = vec![
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
    ];

    let mut reader = IterReader::new(pieces.into_iter());

    let mut buf = [0; 3];

    assert_eq!(0, reader.read(&mut []).unwrap());

    assert_eq!(2, reader.read(&mut buf).unwrap());
    assert_eq!([1,2,0], buf);

    assert_eq!(0, reader.read(&mut buf).unwrap());
    assert_eq!([1,2,0], buf);

    assert_eq!(3, reader.read(&mut buf).unwrap());
    assert_eq!([3,4,5], buf);

    assert_eq!(3, reader.read(&mut buf).unwrap());
    assert_eq!([6,7,8], buf);

    assert_eq!(1, reader.read(&mut buf).unwrap());
    assert_eq!([9,7,8], buf);

    assert_eq!(1, reader.read(&mut buf).unwrap());
    assert_eq!([10,7,8], buf);

    assert_eq!(0, reader.read(&mut buf).unwrap());
    assert_eq!(0, reader.read(&mut buf).unwrap());
}

#[test]
fn test_read_from_chan() {
  use io::Read;

  let mut reader = {
      let (tx, rx) = sync::mpsc::channel();
      tx.send(vec![1, 2]).unwrap();
      tx.send(vec![]).unwrap();
      tx.send(vec![3, 4, 5]).unwrap();
      tx.send(vec![6, 7, 8, 9]).unwrap();
      tx.send(vec![10]).unwrap();

      IterReader::new(rx.into_iter())
  };

  let mut buf = [0; 3];

  assert_eq!(0, reader.read(&mut []).unwrap());

  assert_eq!(2, reader.read(&mut buf).unwrap());
  assert_eq!([1,2,0], buf);

  assert_eq!(0, reader.read(&mut buf).unwrap());
  assert_eq!([1,2,0], buf);

  assert_eq!(3, reader.read(&mut buf).unwrap());
  assert_eq!([3,4,5], buf);

  assert_eq!(3, reader.read(&mut buf).unwrap());
  assert_eq!([6,7,8], buf);

  assert_eq!(1, reader.read(&mut buf).unwrap());
  assert_eq!([9,7,8], buf);

  assert_eq!(1, reader.read(&mut buf).unwrap());
  assert_eq!([10,7,8], buf);

  assert_eq!(0, reader.read(&mut buf).unwrap());
  assert_eq!(0, reader.read(&mut buf).unwrap());
}

#[test]
fn test_bufread() {
    let pieces = vec![
        b"he".to_vec(),
        b"llo wo".to_vec(),
        b"".to_vec(),
        b"rld\n\nhow ".to_vec(),
        b"are you?".to_vec(),
        b"".to_vec(),
    ];

    let lines = vec![
        "hello world\n",
        "\n",
        "how are you?",
        "",
        "",
    ];

    let mut reader = IterReader::new(pieces.into_iter());

    for line in lines {
        let buf = &mut "".to_string();
        assert_eq!(line.len(), reader.read_line(buf).unwrap());
        assert_eq!(line.to_string(), *buf);
    }
}
