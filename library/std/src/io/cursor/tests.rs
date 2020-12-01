use crate::io::prelude::*;
use crate::io::{Cursor, IoSlice, IoSliceMut, SeekFrom};

#[test]
fn test_vec_writer() {
    let mut writer = Vec::new();
    assert_eq!(writer.write(&[0]).unwrap(), 1);
    assert_eq!(writer.write(&[1, 2, 3]).unwrap(), 3);
    assert_eq!(writer.write(&[4, 5, 6, 7]).unwrap(), 4);
    assert_eq!(
        writer
            .write_vectored(&[IoSlice::new(&[]), IoSlice::new(&[8, 9]), IoSlice::new(&[10])],)
            .unwrap(),
        3
    );
    let b: &[_] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(writer, b);
}

#[test]
fn test_mem_writer() {
    let mut writer = Cursor::new(Vec::new());
    assert_eq!(writer.write(&[0]).unwrap(), 1);
    assert_eq!(writer.write(&[1, 2, 3]).unwrap(), 3);
    assert_eq!(writer.write(&[4, 5, 6, 7]).unwrap(), 4);
    assert_eq!(
        writer
            .write_vectored(&[IoSlice::new(&[]), IoSlice::new(&[8, 9]), IoSlice::new(&[10])],)
            .unwrap(),
        3
    );
    let b: &[_] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(&writer.get_ref()[..], b);
}

#[test]
fn test_mem_mut_writer() {
    let mut vec = Vec::new();
    let mut writer = Cursor::new(&mut vec);
    assert_eq!(writer.write(&[0]).unwrap(), 1);
    assert_eq!(writer.write(&[1, 2, 3]).unwrap(), 3);
    assert_eq!(writer.write(&[4, 5, 6, 7]).unwrap(), 4);
    assert_eq!(
        writer
            .write_vectored(&[IoSlice::new(&[]), IoSlice::new(&[8, 9]), IoSlice::new(&[10])],)
            .unwrap(),
        3
    );
    let b: &[_] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(&writer.get_ref()[..], b);
}

#[test]
fn test_box_slice_writer() {
    let mut writer = Cursor::new(vec![0u8; 9].into_boxed_slice());
    assert_eq!(writer.position(), 0);
    assert_eq!(writer.write(&[0]).unwrap(), 1);
    assert_eq!(writer.position(), 1);
    assert_eq!(writer.write(&[1, 2, 3]).unwrap(), 3);
    assert_eq!(writer.write(&[4, 5, 6, 7]).unwrap(), 4);
    assert_eq!(writer.position(), 8);
    assert_eq!(writer.write(&[]).unwrap(), 0);
    assert_eq!(writer.position(), 8);

    assert_eq!(writer.write(&[8, 9]).unwrap(), 1);
    assert_eq!(writer.write(&[10]).unwrap(), 0);
    let b: &[_] = &[0, 1, 2, 3, 4, 5, 6, 7, 8];
    assert_eq!(&**writer.get_ref(), b);
}

#[test]
fn test_box_slice_writer_vectored() {
    let mut writer = Cursor::new(vec![0u8; 9].into_boxed_slice());
    assert_eq!(writer.position(), 0);
    assert_eq!(writer.write_vectored(&[IoSlice::new(&[0])]).unwrap(), 1);
    assert_eq!(writer.position(), 1);
    assert_eq!(
        writer.write_vectored(&[IoSlice::new(&[1, 2, 3]), IoSlice::new(&[4, 5, 6, 7]),]).unwrap(),
        7,
    );
    assert_eq!(writer.position(), 8);
    assert_eq!(writer.write_vectored(&[]).unwrap(), 0);
    assert_eq!(writer.position(), 8);

    assert_eq!(writer.write_vectored(&[IoSlice::new(&[8, 9])]).unwrap(), 1);
    assert_eq!(writer.write_vectored(&[IoSlice::new(&[10])]).unwrap(), 0);
    let b: &[_] = &[0, 1, 2, 3, 4, 5, 6, 7, 8];
    assert_eq!(&**writer.get_ref(), b);
}

#[test]
fn test_buf_writer() {
    let mut buf = [0 as u8; 9];
    {
        let mut writer = Cursor::new(&mut buf[..]);
        assert_eq!(writer.position(), 0);
        assert_eq!(writer.write(&[0]).unwrap(), 1);
        assert_eq!(writer.position(), 1);
        assert_eq!(writer.write(&[1, 2, 3]).unwrap(), 3);
        assert_eq!(writer.write(&[4, 5, 6, 7]).unwrap(), 4);
        assert_eq!(writer.position(), 8);
        assert_eq!(writer.write(&[]).unwrap(), 0);
        assert_eq!(writer.position(), 8);

        assert_eq!(writer.write(&[8, 9]).unwrap(), 1);
        assert_eq!(writer.write(&[10]).unwrap(), 0);
    }
    let b: &[_] = &[0, 1, 2, 3, 4, 5, 6, 7, 8];
    assert_eq!(buf, b);
}

#[test]
fn test_buf_writer_vectored() {
    let mut buf = [0 as u8; 9];
    {
        let mut writer = Cursor::new(&mut buf[..]);
        assert_eq!(writer.position(), 0);
        assert_eq!(writer.write_vectored(&[IoSlice::new(&[0])]).unwrap(), 1);
        assert_eq!(writer.position(), 1);
        assert_eq!(
            writer
                .write_vectored(&[IoSlice::new(&[1, 2, 3]), IoSlice::new(&[4, 5, 6, 7])],)
                .unwrap(),
            7,
        );
        assert_eq!(writer.position(), 8);
        assert_eq!(writer.write_vectored(&[]).unwrap(), 0);
        assert_eq!(writer.position(), 8);

        assert_eq!(writer.write_vectored(&[IoSlice::new(&[8, 9])]).unwrap(), 1);
        assert_eq!(writer.write_vectored(&[IoSlice::new(&[10])]).unwrap(), 0);
    }
    let b: &[_] = &[0, 1, 2, 3, 4, 5, 6, 7, 8];
    assert_eq!(buf, b);
}

#[test]
fn test_buf_writer_seek() {
    let mut buf = [0 as u8; 8];
    {
        let mut writer = Cursor::new(&mut buf[..]);
        assert_eq!(writer.position(), 0);
        assert_eq!(writer.write(&[1]).unwrap(), 1);
        assert_eq!(writer.position(), 1);

        assert_eq!(writer.seek(SeekFrom::Start(2)).unwrap(), 2);
        assert_eq!(writer.position(), 2);
        assert_eq!(writer.write(&[2]).unwrap(), 1);
        assert_eq!(writer.position(), 3);

        assert_eq!(writer.seek(SeekFrom::Current(-2)).unwrap(), 1);
        assert_eq!(writer.position(), 1);
        assert_eq!(writer.write(&[3]).unwrap(), 1);
        assert_eq!(writer.position(), 2);

        assert_eq!(writer.seek(SeekFrom::End(-1)).unwrap(), 7);
        assert_eq!(writer.position(), 7);
        assert_eq!(writer.write(&[4]).unwrap(), 1);
        assert_eq!(writer.position(), 8);
    }
    let b: &[_] = &[1, 3, 2, 0, 0, 0, 0, 4];
    assert_eq!(buf, b);
}

#[test]
fn test_buf_writer_error() {
    let mut buf = [0 as u8; 2];
    let mut writer = Cursor::new(&mut buf[..]);
    assert_eq!(writer.write(&[0]).unwrap(), 1);
    assert_eq!(writer.write(&[0, 0]).unwrap(), 1);
    assert_eq!(writer.write(&[0, 0]).unwrap(), 0);
}

#[test]
fn test_mem_reader() {
    let mut reader = Cursor::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
    let mut buf = [];
    assert_eq!(reader.read(&mut buf).unwrap(), 0);
    assert_eq!(reader.position(), 0);
    let mut buf = [0];
    assert_eq!(reader.read(&mut buf).unwrap(), 1);
    assert_eq!(reader.position(), 1);
    let b: &[_] = &[0];
    assert_eq!(buf, b);
    let mut buf = [0; 4];
    assert_eq!(reader.read(&mut buf).unwrap(), 4);
    assert_eq!(reader.position(), 5);
    let b: &[_] = &[1, 2, 3, 4];
    assert_eq!(buf, b);
    assert_eq!(reader.read(&mut buf).unwrap(), 3);
    let b: &[_] = &[5, 6, 7];
    assert_eq!(&buf[..3], b);
    assert_eq!(reader.read(&mut buf).unwrap(), 0);
}

#[test]
fn test_mem_reader_vectored() {
    let mut reader = Cursor::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
    let mut buf = [];
    assert_eq!(reader.read_vectored(&mut [IoSliceMut::new(&mut buf)]).unwrap(), 0);
    assert_eq!(reader.position(), 0);
    let mut buf = [0];
    assert_eq!(
        reader.read_vectored(&mut [IoSliceMut::new(&mut []), IoSliceMut::new(&mut buf),]).unwrap(),
        1,
    );
    assert_eq!(reader.position(), 1);
    let b: &[_] = &[0];
    assert_eq!(buf, b);
    let mut buf1 = [0; 4];
    let mut buf2 = [0; 4];
    assert_eq!(
        reader
            .read_vectored(&mut [IoSliceMut::new(&mut buf1), IoSliceMut::new(&mut buf2),])
            .unwrap(),
        7,
    );
    let b1: &[_] = &[1, 2, 3, 4];
    let b2: &[_] = &[5, 6, 7];
    assert_eq!(buf1, b1);
    assert_eq!(&buf2[..3], b2);
    assert_eq!(reader.read(&mut buf).unwrap(), 0);
}

#[test]
fn test_boxed_slice_reader() {
    let mut reader = Cursor::new(vec![0, 1, 2, 3, 4, 5, 6, 7].into_boxed_slice());
    let mut buf = [];
    assert_eq!(reader.read(&mut buf).unwrap(), 0);
    assert_eq!(reader.position(), 0);
    let mut buf = [0];
    assert_eq!(reader.read(&mut buf).unwrap(), 1);
    assert_eq!(reader.position(), 1);
    let b: &[_] = &[0];
    assert_eq!(buf, b);
    let mut buf = [0; 4];
    assert_eq!(reader.read(&mut buf).unwrap(), 4);
    assert_eq!(reader.position(), 5);
    let b: &[_] = &[1, 2, 3, 4];
    assert_eq!(buf, b);
    assert_eq!(reader.read(&mut buf).unwrap(), 3);
    let b: &[_] = &[5, 6, 7];
    assert_eq!(&buf[..3], b);
    assert_eq!(reader.read(&mut buf).unwrap(), 0);
}

#[test]
fn test_boxed_slice_reader_vectored() {
    let mut reader = Cursor::new(vec![0, 1, 2, 3, 4, 5, 6, 7].into_boxed_slice());
    let mut buf = [];
    assert_eq!(reader.read_vectored(&mut [IoSliceMut::new(&mut buf)]).unwrap(), 0);
    assert_eq!(reader.position(), 0);
    let mut buf = [0];
    assert_eq!(
        reader.read_vectored(&mut [IoSliceMut::new(&mut []), IoSliceMut::new(&mut buf),]).unwrap(),
        1,
    );
    assert_eq!(reader.position(), 1);
    let b: &[_] = &[0];
    assert_eq!(buf, b);
    let mut buf1 = [0; 4];
    let mut buf2 = [0; 4];
    assert_eq!(
        reader
            .read_vectored(&mut [IoSliceMut::new(&mut buf1), IoSliceMut::new(&mut buf2)],)
            .unwrap(),
        7,
    );
    let b1: &[_] = &[1, 2, 3, 4];
    let b2: &[_] = &[5, 6, 7];
    assert_eq!(buf1, b1);
    assert_eq!(&buf2[..3], b2);
    assert_eq!(reader.read(&mut buf).unwrap(), 0);
}

#[test]
fn read_to_end() {
    let mut reader = Cursor::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
    let mut v = Vec::new();
    reader.read_to_end(&mut v).unwrap();
    assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn test_slice_reader() {
    let in_buf = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let reader = &mut &in_buf[..];
    let mut buf = [];
    assert_eq!(reader.read(&mut buf).unwrap(), 0);
    let mut buf = [0];
    assert_eq!(reader.read(&mut buf).unwrap(), 1);
    assert_eq!(reader.len(), 7);
    let b: &[_] = &[0];
    assert_eq!(&buf[..], b);
    let mut buf = [0; 4];
    assert_eq!(reader.read(&mut buf).unwrap(), 4);
    assert_eq!(reader.len(), 3);
    let b: &[_] = &[1, 2, 3, 4];
    assert_eq!(&buf[..], b);
    assert_eq!(reader.read(&mut buf).unwrap(), 3);
    let b: &[_] = &[5, 6, 7];
    assert_eq!(&buf[..3], b);
    assert_eq!(reader.read(&mut buf).unwrap(), 0);
}

#[test]
fn test_slice_reader_vectored() {
    let in_buf = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let reader = &mut &in_buf[..];
    let mut buf = [];
    assert_eq!(reader.read_vectored(&mut [IoSliceMut::new(&mut buf)]).unwrap(), 0);
    let mut buf = [0];
    assert_eq!(
        reader.read_vectored(&mut [IoSliceMut::new(&mut []), IoSliceMut::new(&mut buf),]).unwrap(),
        1,
    );
    assert_eq!(reader.len(), 7);
    let b: &[_] = &[0];
    assert_eq!(buf, b);
    let mut buf1 = [0; 4];
    let mut buf2 = [0; 4];
    assert_eq!(
        reader
            .read_vectored(&mut [IoSliceMut::new(&mut buf1), IoSliceMut::new(&mut buf2)],)
            .unwrap(),
        7,
    );
    let b1: &[_] = &[1, 2, 3, 4];
    let b2: &[_] = &[5, 6, 7];
    assert_eq!(buf1, b1);
    assert_eq!(&buf2[..3], b2);
    assert_eq!(reader.read(&mut buf).unwrap(), 0);
}

#[test]
fn test_read_exact() {
    let in_buf = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let reader = &mut &in_buf[..];
    let mut buf = [];
    assert!(reader.read_exact(&mut buf).is_ok());
    let mut buf = [8];
    assert!(reader.read_exact(&mut buf).is_ok());
    assert_eq!(buf[0], 0);
    assert_eq!(reader.len(), 7);
    let mut buf = [0, 0, 0, 0, 0, 0, 0];
    assert!(reader.read_exact(&mut buf).is_ok());
    assert_eq!(buf, [1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(reader.len(), 0);
    let mut buf = [0];
    assert!(reader.read_exact(&mut buf).is_err());
}

#[test]
fn test_buf_reader() {
    let in_buf = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let mut reader = Cursor::new(&in_buf[..]);
    let mut buf = [];
    assert_eq!(reader.read(&mut buf).unwrap(), 0);
    assert_eq!(reader.position(), 0);
    let mut buf = [0];
    assert_eq!(reader.read(&mut buf).unwrap(), 1);
    assert_eq!(reader.position(), 1);
    let b: &[_] = &[0];
    assert_eq!(buf, b);
    let mut buf = [0; 4];
    assert_eq!(reader.read(&mut buf).unwrap(), 4);
    assert_eq!(reader.position(), 5);
    let b: &[_] = &[1, 2, 3, 4];
    assert_eq!(buf, b);
    assert_eq!(reader.read(&mut buf).unwrap(), 3);
    let b: &[_] = &[5, 6, 7];
    assert_eq!(&buf[..3], b);
    assert_eq!(reader.read(&mut buf).unwrap(), 0);
}

#[test]
fn seek_past_end() {
    let buf = [0xff];
    let mut r = Cursor::new(&buf[..]);
    assert_eq!(r.seek(SeekFrom::Start(10)).unwrap(), 10);
    assert_eq!(r.read(&mut [0]).unwrap(), 0);

    let mut r = Cursor::new(vec![10]);
    assert_eq!(r.seek(SeekFrom::Start(10)).unwrap(), 10);
    assert_eq!(r.read(&mut [0]).unwrap(), 0);

    let mut buf = [0];
    let mut r = Cursor::new(&mut buf[..]);
    assert_eq!(r.seek(SeekFrom::Start(10)).unwrap(), 10);
    assert_eq!(r.write(&[3]).unwrap(), 0);

    let mut r = Cursor::new(vec![10].into_boxed_slice());
    assert_eq!(r.seek(SeekFrom::Start(10)).unwrap(), 10);
    assert_eq!(r.write(&[3]).unwrap(), 0);
}

#[test]
fn seek_past_i64() {
    let buf = [0xff];
    let mut r = Cursor::new(&buf[..]);
    assert_eq!(r.seek(SeekFrom::Start(6)).unwrap(), 6);
    assert_eq!(r.seek(SeekFrom::Current(0x7ffffffffffffff0)).unwrap(), 0x7ffffffffffffff6);
    assert_eq!(r.seek(SeekFrom::Current(0x10)).unwrap(), 0x8000000000000006);
    assert_eq!(r.seek(SeekFrom::Current(0)).unwrap(), 0x8000000000000006);
    assert!(r.seek(SeekFrom::Current(0x7ffffffffffffffd)).is_err());
    assert_eq!(r.seek(SeekFrom::Current(-0x8000000000000000)).unwrap(), 6);

    let mut r = Cursor::new(vec![10]);
    assert_eq!(r.seek(SeekFrom::Start(6)).unwrap(), 6);
    assert_eq!(r.seek(SeekFrom::Current(0x7ffffffffffffff0)).unwrap(), 0x7ffffffffffffff6);
    assert_eq!(r.seek(SeekFrom::Current(0x10)).unwrap(), 0x8000000000000006);
    assert_eq!(r.seek(SeekFrom::Current(0)).unwrap(), 0x8000000000000006);
    assert!(r.seek(SeekFrom::Current(0x7ffffffffffffffd)).is_err());
    assert_eq!(r.seek(SeekFrom::Current(-0x8000000000000000)).unwrap(), 6);

    let mut buf = [0];
    let mut r = Cursor::new(&mut buf[..]);
    assert_eq!(r.seek(SeekFrom::Start(6)).unwrap(), 6);
    assert_eq!(r.seek(SeekFrom::Current(0x7ffffffffffffff0)).unwrap(), 0x7ffffffffffffff6);
    assert_eq!(r.seek(SeekFrom::Current(0x10)).unwrap(), 0x8000000000000006);
    assert_eq!(r.seek(SeekFrom::Current(0)).unwrap(), 0x8000000000000006);
    assert!(r.seek(SeekFrom::Current(0x7ffffffffffffffd)).is_err());
    assert_eq!(r.seek(SeekFrom::Current(-0x8000000000000000)).unwrap(), 6);

    let mut r = Cursor::new(vec![10].into_boxed_slice());
    assert_eq!(r.seek(SeekFrom::Start(6)).unwrap(), 6);
    assert_eq!(r.seek(SeekFrom::Current(0x7ffffffffffffff0)).unwrap(), 0x7ffffffffffffff6);
    assert_eq!(r.seek(SeekFrom::Current(0x10)).unwrap(), 0x8000000000000006);
    assert_eq!(r.seek(SeekFrom::Current(0)).unwrap(), 0x8000000000000006);
    assert!(r.seek(SeekFrom::Current(0x7ffffffffffffffd)).is_err());
    assert_eq!(r.seek(SeekFrom::Current(-0x8000000000000000)).unwrap(), 6);
}

#[test]
fn seek_before_0() {
    let buf = [0xff];
    let mut r = Cursor::new(&buf[..]);
    assert!(r.seek(SeekFrom::End(-2)).is_err());

    let mut r = Cursor::new(vec![10]);
    assert!(r.seek(SeekFrom::End(-2)).is_err());

    let mut buf = [0];
    let mut r = Cursor::new(&mut buf[..]);
    assert!(r.seek(SeekFrom::End(-2)).is_err());

    let mut r = Cursor::new(vec![10].into_boxed_slice());
    assert!(r.seek(SeekFrom::End(-2)).is_err());
}

#[test]
fn test_seekable_mem_writer() {
    let mut writer = Cursor::new(Vec::<u8>::new());
    assert_eq!(writer.position(), 0);
    assert_eq!(writer.write(&[0]).unwrap(), 1);
    assert_eq!(writer.position(), 1);
    assert_eq!(writer.write(&[1, 2, 3]).unwrap(), 3);
    assert_eq!(writer.write(&[4, 5, 6, 7]).unwrap(), 4);
    assert_eq!(writer.position(), 8);
    let b: &[_] = &[0, 1, 2, 3, 4, 5, 6, 7];
    assert_eq!(&writer.get_ref()[..], b);

    assert_eq!(writer.seek(SeekFrom::Start(0)).unwrap(), 0);
    assert_eq!(writer.position(), 0);
    assert_eq!(writer.write(&[3, 4]).unwrap(), 2);
    let b: &[_] = &[3, 4, 2, 3, 4, 5, 6, 7];
    assert_eq!(&writer.get_ref()[..], b);

    assert_eq!(writer.seek(SeekFrom::Current(1)).unwrap(), 3);
    assert_eq!(writer.write(&[0, 1]).unwrap(), 2);
    let b: &[_] = &[3, 4, 2, 0, 1, 5, 6, 7];
    assert_eq!(&writer.get_ref()[..], b);

    assert_eq!(writer.seek(SeekFrom::End(-1)).unwrap(), 7);
    assert_eq!(writer.write(&[1, 2]).unwrap(), 2);
    let b: &[_] = &[3, 4, 2, 0, 1, 5, 6, 1, 2];
    assert_eq!(&writer.get_ref()[..], b);

    assert_eq!(writer.seek(SeekFrom::End(1)).unwrap(), 10);
    assert_eq!(writer.write(&[1]).unwrap(), 1);
    let b: &[_] = &[3, 4, 2, 0, 1, 5, 6, 1, 2, 0, 1];
    assert_eq!(&writer.get_ref()[..], b);
}

#[test]
fn vec_seek_past_end() {
    let mut r = Cursor::new(Vec::new());
    assert_eq!(r.seek(SeekFrom::Start(10)).unwrap(), 10);
    assert_eq!(r.write(&[3]).unwrap(), 1);
}

#[test]
fn vec_seek_before_0() {
    let mut r = Cursor::new(Vec::new());
    assert!(r.seek(SeekFrom::End(-2)).is_err());
}

#[test]
#[cfg(target_pointer_width = "32")]
fn vec_seek_and_write_past_usize_max() {
    let mut c = Cursor::new(Vec::new());
    c.set_position(usize::MAX as u64 + 1);
    assert!(c.write_all(&[1, 2, 3]).is_err());
}

#[test]
fn test_partial_eq() {
    assert_eq!(Cursor::new(Vec::<u8>::new()), Cursor::new(Vec::<u8>::new()));
}

#[test]
fn test_eq() {
    struct AssertEq<T: Eq>(pub T);

    let _: AssertEq<Cursor<Vec<u8>>> = AssertEq(Cursor::new(Vec::new()));
}

#[allow(dead_code)]
fn const_cursor() {
    const CURSOR: Cursor<&[u8]> = Cursor::new(&[0]);
    const _: &&[u8] = CURSOR.get_ref();
    const _: u64 = CURSOR.position();
}
