/*!

The `ToBytes` and `IterBytes` traits

*/

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use io::Writer;

pub type Cb = fn(buf: &[const u8]) -> bool;

/**
 * A trait to implement in order to make a type hashable;
 * This works in combination with the trait `Hash::Hash`, and
 * may in the future be merged with that trait or otherwise
 * modified when default methods and trait inheritence are
 * completed.
 */
#[cfg(stage0)]
pub trait IterBytes {
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

#[cfg(stage1)]
#[cfg(stage2)]
pub trait IterBytes {
    pure fn iter_bytes(&self, lsb0: bool, f: Cb);
}

#[cfg(stage0)]
impl bool: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(_lsb0: bool, f: Cb) {
        f([
            self as u8
        ]);
    }
}

#[cfg(stage1)]
#[cfg(stage2)]
impl bool: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, _lsb0: bool, f: Cb) {
        f([
            *self as u8
        ]);
    }
}

#[cfg(stage0)]
impl u8: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(_lsb0: bool, f: Cb) {
        f([
            self
        ]);
    }
}

#[cfg(stage1)]
#[cfg(stage2)]
impl u8: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, _lsb0: bool, f: Cb) {
        f([
            *self
        ]);
    }
}

#[cfg(stage0)]
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

#[cfg(stage1)]
#[cfg(stage2)]
impl u16: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
        if lsb0 {
            f([
                *self as u8,
                (*self >> 8) as u8
            ]);
        } else {
            f([
                (*self >> 8) as u8,
                *self as u8
            ]);
        }
    }
}

#[cfg(stage0)]
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

#[cfg(stage1)]
#[cfg(stage2)]
impl u32: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
        if lsb0 {
            f([
                *self as u8,
                (*self >> 8) as u8,
                (*self >> 16) as u8,
                (*self >> 24) as u8,
            ]);
        } else {
            f([
                (*self >> 24) as u8,
                (*self >> 16) as u8,
                (*self >> 8) as u8,
                *self as u8
            ]);
        }
    }
}

#[cfg(stage0)]
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

#[cfg(stage1)]
#[cfg(stage2)]
impl u64: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
        if lsb0 {
            f([
                *self as u8,
                (*self >> 8) as u8,
                (*self >> 16) as u8,
                (*self >> 24) as u8,
                (*self >> 32) as u8,
                (*self >> 40) as u8,
                (*self >> 48) as u8,
                (*self >> 56) as u8
            ]);
        } else {
            f([
                (*self >> 56) as u8,
                (*self >> 48) as u8,
                (*self >> 40) as u8,
                (*self >> 32) as u8,
                (*self >> 24) as u8,
                (*self >> 16) as u8,
                (*self >> 8) as u8,
                *self as u8
            ]);
        }
    }
}

#[cfg(stage0)]
impl i8: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (self as u8).iter_bytes(lsb0, f)
    }
}

#[cfg(stage1)]
#[cfg(stage2)]
impl i8: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (*self as u8).iter_bytes(lsb0, f)
    }
}

#[cfg(stage0)]
impl i16: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (self as u16).iter_bytes(lsb0, f)
    }
}

#[cfg(stage1)]
#[cfg(stage2)]
impl i16: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (*self as u16).iter_bytes(lsb0, f)
    }
}

#[cfg(stage0)]
impl i32: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (self as u32).iter_bytes(lsb0, f)
    }
}

#[cfg(stage1)]
#[cfg(stage2)]
impl i32: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (*self as u32).iter_bytes(lsb0, f)
    }
}

#[cfg(stage0)]
impl i64: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (self as u64).iter_bytes(lsb0, f)
    }
}

#[cfg(stage1)]
#[cfg(stage2)]
impl i64: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (*self as u64).iter_bytes(lsb0, f)
    }
}

#[cfg(stage0)]
impl char: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (self as u32).iter_bytes(lsb0, f)
    }
}

#[cfg(stage1)]
#[cfg(stage2)]
impl char: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (*self as u32).iter_bytes(lsb0, f)
    }
}

#[cfg(target_word_size = "32")]
pub mod x32 {
    #[cfg(stage0)]
    pub impl uint: IterBytes {
        #[inline(always)]
        pure fn iter_bytes(lsb0: bool, f: Cb) {
            (self as u32).iter_bytes(lsb0, f)
        }
    }

    #[cfg(stage1)]
    #[cfg(stage2)]
    pub impl uint: IterBytes {
        #[inline(always)]
        pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
            (*self as u32).iter_bytes(lsb0, f)
        }
    }
}

#[cfg(target_word_size = "64")]
pub mod x64 {
    #[cfg(stage0)]
    pub impl uint: IterBytes {
        #[inline(always)]
        pure fn iter_bytes(lsb0: bool, f: Cb) {
            (self as u64).iter_bytes(lsb0, f)
        }
    }

    #[cfg(stage1)]
    #[cfg(stage2)]
    pub impl uint: IterBytes {
        #[inline(always)]
        pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
            (*self as u64).iter_bytes(lsb0, f)
        }
    }
}

#[cfg(stage0)]
impl int: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (self as uint).iter_bytes(lsb0, f)
    }
}

#[cfg(stage1)]
#[cfg(stage2)]
impl int: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (*self as uint).iter_bytes(lsb0, f)
    }
}

#[cfg(stage0)]
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

#[cfg(stage1)]
#[cfg(stage2)]
impl<A: IterBytes> &[A]: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
        for (*self).each |elt| {
            do elt.iter_bytes(lsb0) |bytes| {
                f(bytes)
            }
        }
    }
}

#[cfg(stage0)]
impl<A: IterBytes, B: IterBytes> (A,B): IterBytes {
  #[inline(always)]
  pure fn iter_bytes(lsb0: bool, f: Cb) {
    let &(ref a, ref b) = &self;
    a.iter_bytes(lsb0, f);
    b.iter_bytes(lsb0, f);
  }
}

#[cfg(stage1)]
#[cfg(stage2)]
impl<A: IterBytes, B: IterBytes> (A,B): IterBytes {
  #[inline(always)]
  pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
    let &(ref a, ref b) = self;
    a.iter_bytes(lsb0, f);
    b.iter_bytes(lsb0, f);
  }
}

#[cfg(stage0)]
impl<A: IterBytes, B: IterBytes, C: IterBytes> (A,B,C): IterBytes {
  #[inline(always)]
  pure fn iter_bytes(lsb0: bool, f: Cb) {
    let &(ref a, ref b, ref c) = &self;
    a.iter_bytes(lsb0, f);
    b.iter_bytes(lsb0, f);
    c.iter_bytes(lsb0, f);
  }
}

#[cfg(stage1)]
#[cfg(stage2)]
impl<A: IterBytes, B: IterBytes, C: IterBytes> (A,B,C): IterBytes {
  #[inline(always)]
  pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
    let &(ref a, ref b, ref c) = self;
    a.iter_bytes(lsb0, f);
    b.iter_bytes(lsb0, f);
    c.iter_bytes(lsb0, f);
  }
}

// Move this to vec, probably.
pure fn borrow<A>(a: &x/[A]) -> &x/[A] {
    a
}

#[cfg(stage0)]
impl<A: IterBytes> ~[A]: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        borrow(self).iter_bytes(lsb0, f)
    }
}

#[cfg(stage1)]
#[cfg(stage2)]
impl<A: IterBytes> ~[A]: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
        borrow(*self).iter_bytes(lsb0, f)
    }
}

#[cfg(stage0)]
impl<A: IterBytes> @[A]: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        borrow(self).iter_bytes(lsb0, f)
    }
}

#[cfg(stage1)]
#[cfg(stage2)]
impl<A: IterBytes> @[A]: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
        borrow(*self).iter_bytes(lsb0, f)
    }
}

pub pure fn iter_bytes_2<A: IterBytes, B: IterBytes>(a: &A, b: &B,
                                            lsb0: bool, z: Cb) {
    let mut flag = true;
    a.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    b.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
}

pub pure fn iter_bytes_3<A: IterBytes,
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

pub pure fn iter_bytes_4<A: IterBytes,
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

pub pure fn iter_bytes_5<A: IterBytes,
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

pub pure fn iter_bytes_6<A: IterBytes,
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

pub pure fn iter_bytes_7<A: IterBytes,
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

#[cfg(stage0)]
impl &str: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(_lsb0: bool, f: Cb) {
        do str::byte_slice(self) |bytes| {
            f(bytes);
        }
    }
}

#[cfg(stage1)]
#[cfg(stage2)]
impl &str: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, _lsb0: bool, f: Cb) {
        do str::byte_slice(*self) |bytes| {
            f(bytes);
        }
    }
}

#[cfg(stage0)]
impl ~str: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(_lsb0: bool, f: Cb) {
        do str::byte_slice(self) |bytes| {
            f(bytes);
        }
    }
}

#[cfg(stage1)]
#[cfg(stage2)]
impl ~str: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, _lsb0: bool, f: Cb) {
        do str::byte_slice(*self) |bytes| {
            f(bytes);
        }
    }
}

#[cfg(stage0)]
impl @str: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(_lsb0: bool, f: Cb) {
        do str::byte_slice(self) |bytes| {
            f(bytes);
        }
    }
}

#[cfg(stage1)]
#[cfg(stage2)]
impl @str: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, _lsb0: bool, f: Cb) {
        do str::byte_slice(*self) |bytes| {
            f(bytes);
        }
    }
}

#[cfg(stage0)]
impl<A: IterBytes> Option<A>: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        match self {
          Some(ref a) => iter_bytes_2(&0u8, a, lsb0, f),
          None => 1u8.iter_bytes(lsb0, f)
        }
    }
}

#[cfg(stage1)]
#[cfg(stage2)]
impl<A: IterBytes> Option<A>: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
        match *self {
          Some(ref a) => iter_bytes_2(&0u8, a, lsb0, f),
          None => 1u8.iter_bytes(lsb0, f)
        }
    }
}

#[cfg(stage0)]
impl<A: IterBytes> &A: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (*self).iter_bytes(lsb0, f);
    }
}

#[cfg(stage1)]
#[cfg(stage2)]
impl<A: IterBytes> &A: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (**self).iter_bytes(lsb0, f);
    }
}

#[cfg(stage0)]
impl<A: IterBytes> @A: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (*self).iter_bytes(lsb0, f);
    }
}

#[cfg(stage1)]
#[cfg(stage2)]
impl<A: IterBytes> @A: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (**self).iter_bytes(lsb0, f);
    }
}

#[cfg(stage0)]
impl<A: IterBytes> ~A: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (*self).iter_bytes(lsb0, f);
    }
}

#[cfg(stage1)]
#[cfg(stage2)]
impl<A: IterBytes> ~A: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (**self).iter_bytes(lsb0, f);
    }
}

#[cfg(stage0)]
impl<A> *const A: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(lsb0: bool, f: Cb) {
        (self as uint).iter_bytes(lsb0, f);
    }
}

// NB: raw-pointer IterBytes does _not_ dereference
// to the target; it just gives you the pointer-bytes.
#[cfg(stage1)]
#[cfg(stage2)]
impl<A> *const A: IterBytes {
    #[inline(always)]
    pure fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (*self as uint).iter_bytes(lsb0, f);
    }
}


trait ToBytes {
    fn to_bytes(&self, lsb0: bool) -> ~[u8];
}

impl<A: IterBytes> A: ToBytes {
    fn to_bytes(&self, lsb0: bool) -> ~[u8] {
        do io::with_bytes_writer |wr| {
            for self.iter_bytes(lsb0) |bytes| {
                wr.write(bytes)
            }
        }
    }
}
