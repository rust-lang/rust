/*!

The `ToBytes` and `IterBytes` traits

*/

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use io::Writer;

type Cb = fn(buf: &[const u8]) -> bool;

/**
 * A trait to implement in order to make a type hashable;
 * This works in combination with the trait `Hash::Hash`, and
 * may in the future be merged with that trait or otherwise
 * modified when default methods and trait inheritence are
 * completed.
 */
trait IterBytes {
    /**
     * Call the provided callback `f` one or more times with
     * byte-slices that should be used when computing a hash
     * value or otherwise "flattening" the structure into
     * a sequence of bytes. The `lsb0` parameter conveys
     * whether the caller is asking for little-endian bytes
     * (`true`) or big-endian (`false`); this should only be
     * relevant in implementations that represent a single
     * multi-byte datum such as a 32 bit integer or 64 bit
     * floating-point value. It can be safely ignored for
     * larger structured types as they are usually processed
     * left-to-right in declaration order, regardless of
     * underlying memory endianness.
     */
    pure fn iter_bytes(lsb0: bool, f: Cb);
}

impl bool: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(_lsb0: bool, f: Cb) {
        f([
            self as u8
        ]);
    }
}

impl u8: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(_lsb0: bool, f: Cb) {
        f([
            self
        ]);
    }
}

impl u16: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        if lsb0 {
            f([
                self as u8,
                (self >> 8) as u8
            ]);
        } else {
            f([
                (self >> 8) as u8,
                self as u8
            ]);
        }
    }
}

impl u32: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        if lsb0 {
            f([
                self as u8,
                (self >> 8) as u8,
                (self >> 16) as u8,
                (self >> 24) as u8,
            ]);
        } else {
            f([
                (self >> 24) as u8,
                (self >> 16) as u8,
                (self >> 8) as u8,
                self as u8
            ]);
        }
    }
}

impl u64: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        if lsb0 {
            f([
                self as u8,
                (self >> 8) as u8,
                (self >> 16) as u8,
                (self >> 24) as u8,
                (self >> 32) as u8,
                (self >> 40) as u8,
                (self >> 48) as u8,
                (self >> 56) as u8
            ]);
        } else {
            f([
                (self >> 56) as u8,
                (self >> 48) as u8,
                (self >> 40) as u8,
                (self >> 32) as u8,
                (self >> 24) as u8,
                (self >> 16) as u8,
                (self >> 8) as u8,
                self as u8
            ]);
        }
    }
}

impl i8: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (self as u8).iter_bytes(lsb0, f)
    }
}

impl i16: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (self as u16).iter_bytes(lsb0, f)
    }
}

impl i32: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (self as u32).iter_bytes(lsb0, f)
    }
}

impl i64: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (self as u64).iter_bytes(lsb0, f)
    }
}

impl char: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (self as u32).iter_bytes(lsb0, f)
    }
}

#[cfg(target_word_size = "32")]
impl uint: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (self as u32).iter_bytes(lsb0, f)
    }
}

#[cfg(target_word_size = "64")]
impl uint: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (self as u64).iter_bytes(lsb0, f)
    }
}

impl int: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (self as uint).iter_bytes(lsb0, f)
    }
}

impl<A: IterBytes> &[A]: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        for self.each |elt| {
            do elt.iter_bytes(lsb0) |bytes| {
                f(bytes)
            }
        }
    }
}

// Move this to vec, probably.
pure fn borrow<A>(a: &x/[A]) -> &x/[A] {
    a
}

impl<A: IterBytes> ~[A]: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        borrow(self).iter_bytes(lsb0, f)
    }
}


impl<A: IterBytes> @[A]: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        borrow(self).iter_bytes(lsb0, f)
    }
}

pure fn iter_bytes_2<A: IterBytes, B: IterBytes>(a: &A, b: &B,
                                            lsb0: bool, z: Cb) {
    let mut flag = true;
    a.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    b.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
}

pure fn iter_bytes_3<A: IterBytes,
                B: IterBytes,
                C: IterBytes>(a: &A, b: &B, c: &C,
                              lsb0: bool, z: Cb) {
    let mut flag = true;
    a.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    b.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    c.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
}

pure fn iter_bytes_4<A: IterBytes,
                B: IterBytes,
                C: IterBytes,
                D: IterBytes>(a: &A, b: &B, c: &C,
                              d: &D,
                              lsb0: bool, z: Cb) {
    let mut flag = true;
    a.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    b.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    c.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    d.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
}

pure fn iter_bytes_5<A: IterBytes,
                B: IterBytes,
                C: IterBytes,
                D: IterBytes,
                E: IterBytes>(a: &A, b: &B, c: &C,
                              d: &D, e: &E,
                              lsb0: bool, z: Cb) {
    let mut flag = true;
    a.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    b.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    c.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    d.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    e.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
}

pure fn iter_bytes_6<A: IterBytes,
                B: IterBytes,
                C: IterBytes,
                D: IterBytes,
                E: IterBytes,
                F: IterBytes>(a: &A, b: &B, c: &C,
                              d: &D, e: &E, f: &F,
                              lsb0: bool, z: Cb) {
    let mut flag = true;
    a.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    b.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    c.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    d.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    e.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    f.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
}

pure fn iter_bytes_7<A: IterBytes,
                B: IterBytes,
                C: IterBytes,
                D: IterBytes,
                E: IterBytes,
                F: IterBytes,
                G: IterBytes>(a: &A, b: &B, c: &C,
                              d: &D, e: &E, f: &F,
                              g: &G,
                              lsb0: bool, z: Cb) {
    let mut flag = true;
    a.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    b.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    c.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    d.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    e.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    f.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    g.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
}

impl &str: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(_lsb0: bool, f: Cb) {
        do str::byte_slice(self) |bytes| {
            f(bytes);
        }
    }
}

impl ~str: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(_lsb0: bool, f: Cb) {
        do str::byte_slice(self) |bytes| {
            f(bytes);
        }
    }
}

impl @str: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(_lsb0: bool, f: Cb) {
        do str::byte_slice(self) |bytes| {
            f(bytes);
        }
    }
}

impl<A: IterBytes> Option<A>: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        match self {
          Some(ref a) => iter_bytes_2(&0u8, a, lsb0, f),
          None => 1u8.iter_bytes(lsb0, f)
        }
    }
}

impl<A: IterBytes> &A: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (*self).iter_bytes(lsb0, f);
    }
}

impl<A: IterBytes> @A: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (*self).iter_bytes(lsb0, f);
    }
}

impl<A: IterBytes> ~A: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (*self).iter_bytes(lsb0, f);
    }
}

// NB: raw-pointer IterBytes does _not_ dereference
// to the target; it just gives you the pointer-bytes.
impl<A> *const A: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (self as uint).iter_bytes(lsb0, f);
    }
}


trait ToBytes {
    fn to_bytes(lsb0: bool) -> ~[u8];
}

impl<A: IterBytes> A: ToBytes {
    fn to_bytes(lsb0: bool) -> ~[u8] {
        do io::with_bytes_writer |wr| {
            for self.iter_bytes(lsb0) |bytes| {
                wr.write(bytes)
            }
        }
    }
}
