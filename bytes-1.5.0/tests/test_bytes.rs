#![warn(rust_2018_idioms)]

use bytes::{Buf, BufMut, Bytes, BytesMut};

use std::usize;

const LONG: &[u8] = b"mary had a little lamb, little lamb, little lamb";
const SHORT: &[u8] = b"hello world";

fn is_sync<T: Sync>() {}
fn is_send<T: Send>() {}

#[test]
fn test_bounds() {
    is_sync::<Bytes>();
    is_sync::<BytesMut>();
    is_send::<Bytes>();
    is_send::<BytesMut>();
}

#[test]
fn test_layout() {
    use std::mem;

    assert_eq!(
        mem::size_of::<Bytes>(),
        mem::size_of::<usize>() * 4,
        "Bytes size should be 4 words",
    );
    assert_eq!(
        mem::size_of::<BytesMut>(),
        mem::size_of::<usize>() * 4,
        "BytesMut should be 4 words",
    );

    assert_eq!(
        mem::size_of::<Bytes>(),
        mem::size_of::<Option<Bytes>>(),
        "Bytes should be same size as Option<Bytes>",
    );

    assert_eq!(
        mem::size_of::<BytesMut>(),
        mem::size_of::<Option<BytesMut>>(),
        "BytesMut should be same size as Option<BytesMut>",
    );
}

#[test]
fn from_slice() {
    let a = Bytes::from(&b"abcdefgh"[..]);
    assert_eq!(a, b"abcdefgh"[..]);
    assert_eq!(a, &b"abcdefgh"[..]);
    assert_eq!(a, Vec::from(&b"abcdefgh"[..]));
    assert_eq!(b"abcdefgh"[..], a);
    assert_eq!(&b"abcdefgh"[..], a);
    assert_eq!(Vec::from(&b"abcdefgh"[..]), a);

    let a = BytesMut::from(&b"abcdefgh"[..]);
    assert_eq!(a, b"abcdefgh"[..]);
    assert_eq!(a, &b"abcdefgh"[..]);
    assert_eq!(a, Vec::from(&b"abcdefgh"[..]));
    assert_eq!(b"abcdefgh"[..], a);
    assert_eq!(&b"abcdefgh"[..], a);
    assert_eq!(Vec::from(&b"abcdefgh"[..]), a);
}

#[test]
fn fmt() {
    let a = format!("{:?}", Bytes::from(&b"abcdefg"[..]));
    let b = "b\"abcdefg\"";

    assert_eq!(a, b);

    let a = format!("{:?}", BytesMut::from(&b"abcdefg"[..]));
    assert_eq!(a, b);
}

#[test]
fn fmt_write() {
    use std::fmt::Write;
    use std::iter::FromIterator;
    let s = String::from_iter((0..10).map(|_| "abcdefg"));

    let mut a = BytesMut::with_capacity(64);
    write!(a, "{}", &s[..64]).unwrap();
    assert_eq!(a, s[..64].as_bytes());

    let mut b = BytesMut::with_capacity(64);
    write!(b, "{}", &s[..32]).unwrap();
    write!(b, "{}", &s[32..64]).unwrap();
    assert_eq!(b, s[..64].as_bytes());

    let mut c = BytesMut::with_capacity(64);
    write!(c, "{}", s).unwrap();
    assert_eq!(c, s[..].as_bytes());
}

#[test]
fn len() {
    let a = Bytes::from(&b"abcdefg"[..]);
    assert_eq!(a.len(), 7);

    let a = BytesMut::from(&b"abcdefg"[..]);
    assert_eq!(a.len(), 7);

    let a = Bytes::from(&b""[..]);
    assert!(a.is_empty());

    let a = BytesMut::from(&b""[..]);
    assert!(a.is_empty());
}

#[test]
fn index() {
    let a = Bytes::from(&b"hello world"[..]);
    assert_eq!(a[0..5], *b"hello");
}

#[test]
fn slice() {
    let a = Bytes::from(&b"hello world"[..]);

    let b = a.slice(3..5);
    assert_eq!(b, b"lo"[..]);

    let b = a.slice(0..0);
    assert_eq!(b, b""[..]);

    let b = a.slice(3..3);
    assert_eq!(b, b""[..]);

    let b = a.slice(a.len()..a.len());
    assert_eq!(b, b""[..]);

    let b = a.slice(..5);
    assert_eq!(b, b"hello"[..]);

    let b = a.slice(3..);
    assert_eq!(b, b"lo world"[..]);
}

#[test]
#[should_panic]
fn slice_oob_1() {
    let a = Bytes::from(&b"hello world"[..]);
    a.slice(5..44);
}

#[test]
#[should_panic]
fn slice_oob_2() {
    let a = Bytes::from(&b"hello world"[..]);
    a.slice(44..49);
}

#[test]
fn split_off() {
    let mut hello = Bytes::from(&b"helloworld"[..]);
    let world = hello.split_off(5);

    assert_eq!(hello, &b"hello"[..]);
    assert_eq!(world, &b"world"[..]);

    let mut hello = BytesMut::from(&b"helloworld"[..]);
    let world = hello.split_off(5);

    assert_eq!(hello, &b"hello"[..]);
    assert_eq!(world, &b"world"[..]);
}

#[test]
#[should_panic]
fn split_off_oob() {
    let mut hello = Bytes::from(&b"helloworld"[..]);
    let _ = hello.split_off(44);
}

#[test]
fn split_off_uninitialized() {
    let mut bytes = BytesMut::with_capacity(1024);
    let other = bytes.split_off(128);

    assert_eq!(bytes.len(), 0);
    assert_eq!(bytes.capacity(), 128);

    assert_eq!(other.len(), 0);
    assert_eq!(other.capacity(), 896);
}

#[test]
fn split_off_to_loop() {
    let s = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

    for i in 0..(s.len() + 1) {
        {
            let mut bytes = Bytes::from(&s[..]);
            let off = bytes.split_off(i);
            assert_eq!(i, bytes.len());
            let mut sum = Vec::new();
            sum.extend(bytes.iter());
            sum.extend(off.iter());
            assert_eq!(&s[..], &sum[..]);
        }
        {
            let mut bytes = BytesMut::from(&s[..]);
            let off = bytes.split_off(i);
            assert_eq!(i, bytes.len());
            let mut sum = Vec::new();
            sum.extend(&bytes);
            sum.extend(&off);
            assert_eq!(&s[..], &sum[..]);
        }
        {
            let mut bytes = Bytes::from(&s[..]);
            let off = bytes.split_to(i);
            assert_eq!(i, off.len());
            let mut sum = Vec::new();
            sum.extend(off.iter());
            sum.extend(bytes.iter());
            assert_eq!(&s[..], &sum[..]);
        }
        {
            let mut bytes = BytesMut::from(&s[..]);
            let off = bytes.split_to(i);
            assert_eq!(i, off.len());
            let mut sum = Vec::new();
            sum.extend(&off);
            sum.extend(&bytes);
            assert_eq!(&s[..], &sum[..]);
        }
    }
}

#[test]
fn split_to_1() {
    // Static
    let mut a = Bytes::from_static(SHORT);
    let b = a.split_to(4);

    assert_eq!(SHORT[4..], a);
    assert_eq!(SHORT[..4], b);

    // Allocated
    let mut a = Bytes::copy_from_slice(LONG);
    let b = a.split_to(4);

    assert_eq!(LONG[4..], a);
    assert_eq!(LONG[..4], b);

    let mut a = Bytes::copy_from_slice(LONG);
    let b = a.split_to(30);

    assert_eq!(LONG[30..], a);
    assert_eq!(LONG[..30], b);
}

#[test]
fn split_to_2() {
    let mut a = Bytes::from(LONG);
    assert_eq!(LONG, a);

    let b = a.split_to(1);

    assert_eq!(LONG[1..], a);
    drop(b);
}

#[test]
#[should_panic]
fn split_to_oob() {
    let mut hello = Bytes::from(&b"helloworld"[..]);
    let _ = hello.split_to(33);
}

#[test]
#[should_panic]
fn split_to_oob_mut() {
    let mut hello = BytesMut::from(&b"helloworld"[..]);
    let _ = hello.split_to(33);
}

#[test]
#[should_panic]
fn split_to_uninitialized() {
    let mut bytes = BytesMut::with_capacity(1024);
    let _other = bytes.split_to(128);
}

#[test]
fn split_off_to_at_gt_len() {
    fn make_bytes() -> Bytes {
        let mut bytes = BytesMut::with_capacity(100);
        bytes.put_slice(&[10, 20, 30, 40]);
        bytes.freeze()
    }

    use std::panic;

    let _ = make_bytes().split_to(4);
    let _ = make_bytes().split_off(4);

    assert!(panic::catch_unwind(move || {
        let _ = make_bytes().split_to(5);
    })
    .is_err());

    assert!(panic::catch_unwind(move || {
        let _ = make_bytes().split_off(5);
    })
    .is_err());
}

#[test]
fn truncate() {
    let s = &b"helloworld"[..];
    let mut hello = Bytes::from(s);
    hello.truncate(15);
    assert_eq!(hello, s);
    hello.truncate(10);
    assert_eq!(hello, s);
    hello.truncate(5);
    assert_eq!(hello, "hello");
}

#[test]
fn freeze_clone_shared() {
    let s = &b"abcdefgh"[..];
    let b = BytesMut::from(s).split().freeze();
    assert_eq!(b, s);
    let c = b.clone();
    assert_eq!(c, s);
}

#[test]
fn freeze_clone_unique() {
    let s = &b"abcdefgh"[..];
    let b = BytesMut::from(s).freeze();
    assert_eq!(b, s);
    let c = b.clone();
    assert_eq!(c, s);
}

#[test]
fn freeze_after_advance() {
    let s = &b"abcdefgh"[..];
    let mut b = BytesMut::from(s);
    b.advance(1);
    assert_eq!(b, s[1..]);
    let b = b.freeze();
    // Verify fix for #352. Previously, freeze would ignore the start offset
    // for BytesMuts in Vec mode.
    assert_eq!(b, s[1..]);
}

#[test]
fn freeze_after_advance_arc() {
    let s = &b"abcdefgh"[..];
    let mut b = BytesMut::from(s);
    // Make b Arc
    let _ = b.split_to(0);
    b.advance(1);
    assert_eq!(b, s[1..]);
    let b = b.freeze();
    assert_eq!(b, s[1..]);
}

#[test]
fn freeze_after_split_to() {
    let s = &b"abcdefgh"[..];
    let mut b = BytesMut::from(s);
    let _ = b.split_to(1);
    assert_eq!(b, s[1..]);
    let b = b.freeze();
    assert_eq!(b, s[1..]);
}

#[test]
fn freeze_after_truncate() {
    let s = &b"abcdefgh"[..];
    let mut b = BytesMut::from(s);
    b.truncate(7);
    assert_eq!(b, s[..7]);
    let b = b.freeze();
    assert_eq!(b, s[..7]);
}

#[test]
fn freeze_after_truncate_arc() {
    let s = &b"abcdefgh"[..];
    let mut b = BytesMut::from(s);
    // Make b Arc
    let _ = b.split_to(0);
    b.truncate(7);
    assert_eq!(b, s[..7]);
    let b = b.freeze();
    assert_eq!(b, s[..7]);
}

#[test]
fn freeze_after_split_off() {
    let s = &b"abcdefgh"[..];
    let mut b = BytesMut::from(s);
    let _ = b.split_off(7);
    assert_eq!(b, s[..7]);
    let b = b.freeze();
    assert_eq!(b, s[..7]);
}

#[test]
fn fns_defined_for_bytes_mut() {
    let mut bytes = BytesMut::from(&b"hello world"[..]);

    let _ = bytes.as_ptr();
    let _ = bytes.as_mut_ptr();

    // Iterator
    let v: Vec<u8> = bytes.as_ref().iter().cloned().collect();
    assert_eq!(&v[..], bytes);
}

#[test]
fn reserve_convert() {
    // Vec -> Vec
    let mut bytes = BytesMut::from(LONG);
    bytes.reserve(64);
    assert_eq!(bytes.capacity(), LONG.len() + 64);

    // Arc -> Vec
    let mut bytes = BytesMut::from(LONG);
    let a = bytes.split_to(30);

    bytes.reserve(128);
    assert!(bytes.capacity() >= bytes.len() + 128);

    drop(a);
}

#[test]
fn reserve_growth() {
    let mut bytes = BytesMut::with_capacity(64);
    bytes.put("hello world".as_bytes());
    let _ = bytes.split();

    bytes.reserve(65);
    assert_eq!(bytes.capacity(), 117);
}

#[test]
fn reserve_allocates_at_least_original_capacity() {
    let mut bytes = BytesMut::with_capacity(1024);

    for i in 0..1020 {
        bytes.put_u8(i as u8);
    }

    let _other = bytes.split();

    bytes.reserve(16);
    assert_eq!(bytes.capacity(), 1024);
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn reserve_max_original_capacity_value() {
    const SIZE: usize = 128 * 1024;

    let mut bytes = BytesMut::with_capacity(SIZE);

    for _ in 0..SIZE {
        bytes.put_u8(0u8);
    }

    let _other = bytes.split();

    bytes.reserve(16);
    assert_eq!(bytes.capacity(), 64 * 1024);
}

#[test]
fn reserve_vec_recycling() {
    let mut bytes = BytesMut::with_capacity(16);
    assert_eq!(bytes.capacity(), 16);
    let addr = bytes.as_ptr() as usize;
    bytes.put("0123456789012345".as_bytes());
    assert_eq!(bytes.as_ptr() as usize, addr);
    bytes.advance(10);
    assert_eq!(bytes.capacity(), 6);
    bytes.reserve(8);
    assert_eq!(bytes.capacity(), 16);
    assert_eq!(bytes.as_ptr() as usize, addr);
}

#[test]
fn reserve_in_arc_unique_does_not_overallocate() {
    let mut bytes = BytesMut::with_capacity(1000);
    let _ = bytes.split();

    // now bytes is Arc and refcount == 1

    assert_eq!(1000, bytes.capacity());
    bytes.reserve(2001);
    assert_eq!(2001, bytes.capacity());
}

#[test]
fn reserve_in_arc_unique_doubles() {
    let mut bytes = BytesMut::with_capacity(1000);
    let _ = bytes.split();

    // now bytes is Arc and refcount == 1

    assert_eq!(1000, bytes.capacity());
    bytes.reserve(1001);
    assert_eq!(2000, bytes.capacity());
}

#[test]
fn reserve_in_arc_unique_does_not_overallocate_after_split() {
    let mut bytes = BytesMut::from(LONG);
    let orig_capacity = bytes.capacity();
    drop(bytes.split_off(LONG.len() / 2));

    // now bytes is Arc and refcount == 1

    let new_capacity = bytes.capacity();
    bytes.reserve(orig_capacity - new_capacity);
    assert_eq!(bytes.capacity(), orig_capacity);
}

#[test]
fn reserve_in_arc_unique_does_not_overallocate_after_multiple_splits() {
    let mut bytes = BytesMut::from(LONG);
    let orig_capacity = bytes.capacity();
    for _ in 0..10 {
        drop(bytes.split_off(LONG.len() / 2));

        // now bytes is Arc and refcount == 1

        let new_capacity = bytes.capacity();
        bytes.reserve(orig_capacity - new_capacity);
    }
    assert_eq!(bytes.capacity(), orig_capacity);
}

#[test]
fn reserve_in_arc_nonunique_does_not_overallocate() {
    let mut bytes = BytesMut::with_capacity(1000);
    let _copy = bytes.split();

    // now bytes is Arc and refcount == 2

    assert_eq!(1000, bytes.capacity());
    bytes.reserve(2001);
    assert_eq!(2001, bytes.capacity());
}

/// This function tests `BytesMut::reserve_inner`, where `BytesMut` holds
/// a unique reference to the shared vector and decide to reuse it
/// by reallocating the `Vec`.
#[test]
fn reserve_shared_reuse() {
    let mut bytes = BytesMut::with_capacity(1000);
    bytes.put_slice(b"Hello, World!");
    drop(bytes.split());

    bytes.put_slice(b"!123ex123,sadchELLO,_wORLD!");
    // Use split_off so that v.capacity() - self.cap != off
    drop(bytes.split_off(9));
    assert_eq!(&*bytes, b"!123ex123");

    bytes.reserve(2000);
    assert_eq!(&*bytes, b"!123ex123");
    assert_eq!(bytes.capacity(), 2009);
}

#[test]
fn extend_mut() {
    let mut bytes = BytesMut::with_capacity(0);
    bytes.extend(LONG);
    assert_eq!(*bytes, LONG[..]);
}

#[test]
fn extend_from_slice_mut() {
    for &i in &[3, 34] {
        let mut bytes = BytesMut::new();
        bytes.extend_from_slice(&LONG[..i]);
        bytes.extend_from_slice(&LONG[i..]);
        assert_eq!(LONG[..], *bytes);
    }
}

#[test]
fn extend_mut_from_bytes() {
    let mut bytes = BytesMut::with_capacity(0);
    bytes.extend([Bytes::from(LONG)]);
    assert_eq!(*bytes, LONG[..]);
}

#[test]
fn extend_mut_without_size_hint() {
    let mut bytes = BytesMut::with_capacity(0);
    let mut long_iter = LONG.iter();

    // Use iter::from_fn since it doesn't know a size_hint
    bytes.extend(std::iter::from_fn(|| long_iter.next()));
    assert_eq!(*bytes, LONG[..]);
}

#[test]
fn from_static() {
    let mut a = Bytes::from_static(b"ab");
    let b = a.split_off(1);

    assert_eq!(a, b"a"[..]);
    assert_eq!(b, b"b"[..]);
}

#[test]
fn advance_static() {
    let mut a = Bytes::from_static(b"hello world");
    a.advance(6);
    assert_eq!(a, &b"world"[..]);
}

#[test]
fn advance_vec() {
    let mut a = Bytes::from(b"hello world boooo yah world zomg wat wat".to_vec());
    a.advance(16);
    assert_eq!(a, b"o yah world zomg wat wat"[..]);

    a.advance(4);
    assert_eq!(a, b"h world zomg wat wat"[..]);

    a.advance(6);
    assert_eq!(a, b"d zomg wat wat"[..]);
}

#[test]
fn advance_bytes_mut() {
    let mut a = BytesMut::from("hello world boooo yah world zomg wat wat");
    a.advance(16);
    assert_eq!(a, b"o yah world zomg wat wat"[..]);

    a.advance(4);
    assert_eq!(a, b"h world zomg wat wat"[..]);

    // Reserve some space.
    a.reserve(1024);
    assert_eq!(a, b"h world zomg wat wat"[..]);

    a.advance(6);
    assert_eq!(a, b"d zomg wat wat"[..]);
}

#[test]
#[should_panic]
fn advance_past_len() {
    let mut a = BytesMut::from("hello world");
    a.advance(20);
}

#[test]
// Only run these tests on little endian systems. CI uses qemu for testing
// big endian... and qemu doesn't really support threading all that well.
#[cfg(any(miri, target_endian = "little"))]
fn stress() {
    // Tests promoting a buffer from a vec -> shared in a concurrent situation
    use std::sync::{Arc, Barrier};
    use std::thread;

    const THREADS: usize = 8;
    const ITERS: usize = if cfg!(miri) { 100 } else { 1_000 };

    for i in 0..ITERS {
        let data = [i as u8; 256];
        let buf = Arc::new(Bytes::copy_from_slice(&data[..]));

        let barrier = Arc::new(Barrier::new(THREADS));
        let mut joins = Vec::with_capacity(THREADS);

        for _ in 0..THREADS {
            let c = barrier.clone();
            let buf = buf.clone();

            joins.push(thread::spawn(move || {
                c.wait();
                let buf: Bytes = (*buf).clone();
                drop(buf);
            }));
        }

        for th in joins {
            th.join().unwrap();
        }

        assert_eq!(*buf, data[..]);
    }
}

#[test]
fn partial_eq_bytesmut() {
    let bytes = Bytes::from(&b"The quick red fox"[..]);
    let bytesmut = BytesMut::from(&b"The quick red fox"[..]);
    assert!(bytes == bytesmut);
    assert!(bytesmut == bytes);
    let bytes2 = Bytes::from(&b"Jumped over the lazy brown dog"[..]);
    assert!(bytes2 != bytesmut);
    assert!(bytesmut != bytes2);
}

/*
#[test]
fn bytes_unsplit_basic() {
    let buf = Bytes::from(&b"aaabbbcccddd"[..]);

    let splitted = buf.split_off(6);
    assert_eq!(b"aaabbb", &buf[..]);
    assert_eq!(b"cccddd", &splitted[..]);

    buf.unsplit(splitted);
    assert_eq!(b"aaabbbcccddd", &buf[..]);
}

#[test]
fn bytes_unsplit_empty_other() {
    let buf = Bytes::from(&b"aaabbbcccddd"[..]);

    // empty other
    let other = Bytes::new();

    buf.unsplit(other);
    assert_eq!(b"aaabbbcccddd", &buf[..]);
}

#[test]
fn bytes_unsplit_empty_self() {
    // empty self
    let mut buf = Bytes::new();

    let mut other = Bytes::with_capacity(64);
    other.extend_from_slice(b"aaabbbcccddd");

    buf.unsplit(other);
    assert_eq!(b"aaabbbcccddd", &buf[..]);
}

#[test]
fn bytes_unsplit_arc_different() {
    let mut buf = Bytes::with_capacity(64);
    buf.extend_from_slice(b"aaaabbbbeeee");

    buf.split_off(8); //arc

    let mut buf2 = Bytes::with_capacity(64);
    buf2.extend_from_slice(b"ccccddddeeee");

    buf2.split_off(8); //arc

    buf.unsplit(buf2);
    assert_eq!(b"aaaabbbbccccdddd", &buf[..]);
}

#[test]
fn bytes_unsplit_arc_non_contiguous() {
    let mut buf = Bytes::with_capacity(64);
    buf.extend_from_slice(b"aaaabbbbeeeeccccdddd");

    let mut buf2 = buf.split_off(8); //arc

    let buf3 = buf2.split_off(4); //arc

    buf.unsplit(buf3);
    assert_eq!(b"aaaabbbbccccdddd", &buf[..]);
}

#[test]
fn bytes_unsplit_two_split_offs() {
    let mut buf = Bytes::with_capacity(64);
    buf.extend_from_slice(b"aaaabbbbccccdddd");

    let mut buf2 = buf.split_off(8); //arc
    let buf3 = buf2.split_off(4); //arc

    buf2.unsplit(buf3);
    buf.unsplit(buf2);
    assert_eq!(b"aaaabbbbccccdddd", &buf[..]);
}

#[test]
fn bytes_unsplit_overlapping_references() {
    let mut buf = Bytes::with_capacity(64);
    buf.extend_from_slice(b"abcdefghijklmnopqrstuvwxyz");
    let mut buf0010 = buf.slice(0..10);
    let buf1020 = buf.slice(10..20);
    let buf0515 = buf.slice(5..15);
    buf0010.unsplit(buf1020);
    assert_eq!(b"abcdefghijklmnopqrst", &buf0010[..]);
    assert_eq!(b"fghijklmno", &buf0515[..]);
}
*/

#[test]
fn bytes_mut_unsplit_basic() {
    let mut buf = BytesMut::with_capacity(64);
    buf.extend_from_slice(b"aaabbbcccddd");

    let splitted = buf.split_off(6);
    assert_eq!(b"aaabbb", &buf[..]);
    assert_eq!(b"cccddd", &splitted[..]);

    buf.unsplit(splitted);
    assert_eq!(b"aaabbbcccddd", &buf[..]);
}

#[test]
fn bytes_mut_unsplit_empty_other() {
    let mut buf = BytesMut::with_capacity(64);
    buf.extend_from_slice(b"aaabbbcccddd");

    // empty other
    let other = BytesMut::new();

    buf.unsplit(other);
    assert_eq!(b"aaabbbcccddd", &buf[..]);
}

#[test]
fn bytes_mut_unsplit_empty_self() {
    // empty self
    let mut buf = BytesMut::new();

    let mut other = BytesMut::with_capacity(64);
    other.extend_from_slice(b"aaabbbcccddd");

    buf.unsplit(other);
    assert_eq!(b"aaabbbcccddd", &buf[..]);
}

#[test]
fn bytes_mut_unsplit_other_keeps_capacity() {
    let mut buf = BytesMut::with_capacity(64);
    buf.extend_from_slice(b"aabb");

    // non empty other created "from" buf
    let mut other = buf.split_off(buf.len());
    other.extend_from_slice(b"ccddee");
    buf.unsplit(other);

    assert_eq!(buf.capacity(), 64);
}

#[test]
fn bytes_mut_unsplit_empty_other_keeps_capacity() {
    let mut buf = BytesMut::with_capacity(64);
    buf.extend_from_slice(b"aabbccddee");

    // empty other created "from" buf
    let other = buf.split_off(buf.len());
    buf.unsplit(other);

    assert_eq!(buf.capacity(), 64);
}

#[test]
fn bytes_mut_unsplit_arc_different() {
    let mut buf = BytesMut::with_capacity(64);
    buf.extend_from_slice(b"aaaabbbbeeee");

    let _ = buf.split_off(8); //arc

    let mut buf2 = BytesMut::with_capacity(64);
    buf2.extend_from_slice(b"ccccddddeeee");

    let _ = buf2.split_off(8); //arc

    buf.unsplit(buf2);
    assert_eq!(b"aaaabbbbccccdddd", &buf[..]);
}

#[test]
fn bytes_mut_unsplit_arc_non_contiguous() {
    let mut buf = BytesMut::with_capacity(64);
    buf.extend_from_slice(b"aaaabbbbeeeeccccdddd");

    let mut buf2 = buf.split_off(8); //arc

    let buf3 = buf2.split_off(4); //arc

    buf.unsplit(buf3);
    assert_eq!(b"aaaabbbbccccdddd", &buf[..]);
}

#[test]
fn bytes_mut_unsplit_two_split_offs() {
    let mut buf = BytesMut::with_capacity(64);
    buf.extend_from_slice(b"aaaabbbbccccdddd");

    let mut buf2 = buf.split_off(8); //arc
    let buf3 = buf2.split_off(4); //arc

    buf2.unsplit(buf3);
    buf.unsplit(buf2);
    assert_eq!(b"aaaabbbbccccdddd", &buf[..]);
}

#[test]
fn from_iter_no_size_hint() {
    use std::iter;

    let mut expect = vec![];

    let actual: Bytes = iter::repeat(b'x')
        .scan(100, |cnt, item| {
            if *cnt >= 1 {
                *cnt -= 1;
                expect.push(item);
                Some(item)
            } else {
                None
            }
        })
        .collect();

    assert_eq!(&actual[..], &expect[..]);
}

fn test_slice_ref(bytes: &Bytes, start: usize, end: usize, expected: &[u8]) {
    let slice = &(bytes.as_ref()[start..end]);
    let sub = bytes.slice_ref(slice);
    assert_eq!(&sub[..], expected);
}

#[test]
fn slice_ref_works() {
    let bytes = Bytes::from(&b"012345678"[..]);

    test_slice_ref(&bytes, 0, 0, b"");
    test_slice_ref(&bytes, 0, 3, b"012");
    test_slice_ref(&bytes, 2, 6, b"2345");
    test_slice_ref(&bytes, 7, 9, b"78");
    test_slice_ref(&bytes, 9, 9, b"");
}

#[test]
fn slice_ref_empty() {
    let bytes = Bytes::from(&b""[..]);
    let slice = &(bytes.as_ref()[0..0]);

    let sub = bytes.slice_ref(slice);
    assert_eq!(&sub[..], b"");
}

#[test]
fn slice_ref_empty_subslice() {
    let bytes = Bytes::from(&b"abcde"[..]);
    let subbytes = bytes.slice(0..0);
    let slice = &subbytes[..];
    // The `slice` object is derived from the original `bytes` object
    // so `slice_ref` should work.
    assert_eq!(Bytes::new(), bytes.slice_ref(slice));
}

#[test]
#[should_panic]
fn slice_ref_catches_not_a_subset() {
    let bytes = Bytes::from(&b"012345678"[..]);
    let slice = &b"012345"[0..4];

    bytes.slice_ref(slice);
}

#[test]
fn slice_ref_not_an_empty_subset() {
    let bytes = Bytes::from(&b"012345678"[..]);
    let slice = &b""[0..0];

    assert_eq!(Bytes::new(), bytes.slice_ref(slice));
}

#[test]
fn empty_slice_ref_not_an_empty_subset() {
    let bytes = Bytes::new();
    let slice = &b"some other slice"[0..0];

    assert_eq!(Bytes::new(), bytes.slice_ref(slice));
}

#[test]
fn bytes_buf_mut_advance() {
    let mut bytes = BytesMut::with_capacity(1024);

    unsafe {
        let ptr = bytes.chunk_mut().as_mut_ptr();
        assert_eq!(1024, bytes.chunk_mut().len());

        bytes.advance_mut(10);

        let next = bytes.chunk_mut().as_mut_ptr();
        assert_eq!(1024 - 10, bytes.chunk_mut().len());
        assert_eq!(ptr.offset(10), next);

        // advance to the end
        bytes.advance_mut(1024 - 10);

        // The buffer size is doubled
        assert_eq!(1024, bytes.chunk_mut().len());
    }
}

#[test]
fn bytes_buf_mut_reuse_when_fully_consumed() {
    use bytes::{Buf, BytesMut};
    let mut buf = BytesMut::new();
    buf.reserve(8192);
    buf.extend_from_slice(&[0u8; 100][..]);

    let p = &buf[0] as *const u8;
    buf.advance(100);

    buf.reserve(8192);
    buf.extend_from_slice(b" ");

    assert_eq!(&buf[0] as *const u8, p);
}

#[test]
#[should_panic]
fn bytes_reserve_overflow() {
    let mut bytes = BytesMut::with_capacity(1024);
    bytes.put_slice(b"hello world");

    bytes.reserve(usize::MAX);
}

#[test]
fn bytes_with_capacity_but_empty() {
    // See https://github.com/tokio-rs/bytes/issues/340
    let vec = Vec::with_capacity(1);
    let _ = Bytes::from(vec);
}

#[test]
fn bytes_put_bytes() {
    let mut bytes = BytesMut::new();
    bytes.put_u8(17);
    bytes.put_bytes(19, 2);
    assert_eq!([17, 19, 19], bytes.as_ref());
}

#[test]
fn box_slice_empty() {
    // See https://github.com/tokio-rs/bytes/issues/340
    let empty: Box<[u8]> = Default::default();
    let b = Bytes::from(empty);
    assert!(b.is_empty());
}

#[test]
fn bytes_into_vec() {
    // Test kind == KIND_VEC
    let content = b"helloworld";

    let mut bytes = BytesMut::new();
    bytes.put_slice(content);

    let vec: Vec<u8> = bytes.into();
    assert_eq!(&vec, content);

    // Test kind == KIND_ARC, shared.is_unique() == True
    let mut bytes = BytesMut::new();
    bytes.put_slice(b"abcdewe23");
    bytes.put_slice(content);

    // Overwrite the bytes to make sure only one reference to the underlying
    // Vec exists.
    bytes = bytes.split_off(9);

    let vec: Vec<u8> = bytes.into();
    assert_eq!(&vec, content);

    // Test kind == KIND_ARC, shared.is_unique() == False
    let prefix = b"abcdewe23";

    let mut bytes = BytesMut::new();
    bytes.put_slice(prefix);
    bytes.put_slice(content);

    let vec: Vec<u8> = bytes.split_off(prefix.len()).into();
    assert_eq!(&vec, content);

    let vec: Vec<u8> = bytes.into();
    assert_eq!(&vec, prefix);
}

#[test]
fn test_bytes_into_vec() {
    // Test STATIC_VTABLE.to_vec
    let bs = b"1b23exfcz3r";
    let vec: Vec<u8> = Bytes::from_static(bs).into();
    assert_eq!(&*vec, bs);

    // Test bytes_mut.SHARED_VTABLE.to_vec impl
    eprintln!("1");
    let mut bytes_mut: BytesMut = bs[..].into();

    // Set kind to KIND_ARC so that after freeze, Bytes will use bytes_mut.SHARED_VTABLE
    eprintln!("2");
    drop(bytes_mut.split_off(bs.len()));

    eprintln!("3");
    let b1 = bytes_mut.freeze();
    eprintln!("4");
    let b2 = b1.clone();

    eprintln!("{:#?}", (&*b1).as_ptr());

    // shared.is_unique() = False
    eprintln!("5");
    assert_eq!(&*Vec::from(b2), bs);

    // shared.is_unique() = True
    eprintln!("6");
    assert_eq!(&*Vec::from(b1), bs);

    // Test bytes_mut.SHARED_VTABLE.to_vec impl where offset != 0
    let mut bytes_mut1: BytesMut = bs[..].into();
    let bytes_mut2 = bytes_mut1.split_off(9);

    let b1 = bytes_mut1.freeze();
    let b2 = bytes_mut2.freeze();

    assert_eq!(Vec::from(b2), bs[9..]);
    assert_eq!(Vec::from(b1), bs[..9]);
}

#[test]
fn test_bytes_into_vec_promotable_even() {
    let vec = vec![33u8; 1024];

    // Test cases where kind == KIND_VEC
    let b1 = Bytes::from(vec.clone());
    assert_eq!(Vec::from(b1), vec);

    // Test cases where kind == KIND_ARC, ref_cnt == 1
    let b1 = Bytes::from(vec.clone());
    drop(b1.clone());
    assert_eq!(Vec::from(b1), vec);

    // Test cases where kind == KIND_ARC, ref_cnt == 2
    let b1 = Bytes::from(vec.clone());
    let b2 = b1.clone();
    assert_eq!(Vec::from(b1), vec);

    // Test cases where vtable = SHARED_VTABLE, kind == KIND_ARC, ref_cnt == 1
    assert_eq!(Vec::from(b2), vec);

    // Test cases where offset != 0
    let mut b1 = Bytes::from(vec.clone());
    let b2 = b1.split_off(20);

    assert_eq!(Vec::from(b2), vec[20..]);
    assert_eq!(Vec::from(b1), vec[..20]);
}

#[test]
fn test_bytes_vec_conversion() {
    let mut vec = Vec::with_capacity(10);
    vec.extend(b"abcdefg");
    let b = Bytes::from(vec);
    let v = Vec::from(b);
    assert_eq!(v.len(), 7);
    assert_eq!(v.capacity(), 10);

    let mut b = Bytes::from(v);
    b.advance(1);
    let v = Vec::from(b);
    assert_eq!(v.len(), 6);
    assert_eq!(v.capacity(), 10);
    assert_eq!(v.as_slice(), b"bcdefg");
}

#[test]
fn test_bytes_mut_conversion() {
    let mut b1 = BytesMut::with_capacity(10);
    b1.extend(b"abcdefg");
    let b2 = Bytes::from(b1);
    let v = Vec::from(b2);
    assert_eq!(v.len(), 7);
    assert_eq!(v.capacity(), 10);

    let mut b = Bytes::from(v);
    b.advance(1);
    let v = Vec::from(b);
    assert_eq!(v.len(), 6);
    assert_eq!(v.capacity(), 10);
    assert_eq!(v.as_slice(), b"bcdefg");
}

#[test]
fn test_bytes_capacity_len() {
    for cap in 0..100 {
        for len in 0..=cap {
            let mut v = Vec::with_capacity(cap);
            v.resize(len, 0);
            let _ = Bytes::from(v);
        }
    }
}
