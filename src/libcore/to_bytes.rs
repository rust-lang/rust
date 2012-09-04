// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

import io::Writer;

type Cb = fn(buf: &[const u8]) -> bool;

trait IterBytes {
    fn iter_le_bytes(f: Cb);
    fn iter_be_bytes(f: Cb);
}

impl u8: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) {
        f([
            self,
        ]);
    }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) {
        f([
            self as u8
        ]);
    }
}

impl u16: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) {
        f([
            self as u8,
            (self >> 8) as u8
        ]);
    }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) {
        f([
            (self >> 8) as u8,
            self as u8
        ]);
    }
}

impl u32: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) {
        f([
            self as u8,
            (self >> 8) as u8,
            (self >> 16) as u8,
            (self >> 24) as u8,
        ]);
    }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) {
        f([
            (self >> 24) as u8,
            (self >> 16) as u8,
            (self >> 8) as u8,
            self as u8
        ]);
    }
}

impl u64: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) {
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
    }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) {
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

impl i8: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) { (self as u8).iter_le_bytes(f) }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) { (self as u8).iter_be_bytes(f) }
}

impl i16: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) { (self as u16).iter_le_bytes(f) }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) { (self as u16).iter_be_bytes(f) }
}

impl i32: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) { (self as u32).iter_le_bytes(f) }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) { (self as u32).iter_be_bytes(f) }
}

impl i64: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) { (self as u64).iter_le_bytes(f) }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) { (self as u64).iter_be_bytes(f) }
}

#[cfg(target_word_size = "32")]
impl uint: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) { (self as u32).iter_le_bytes(f) }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) { (self as u32).iter_be_bytes(f) }
}

#[cfg(target_word_size = "64")]
impl uint: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) { (self as u64).iter_le_bytes(f) }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) { (self as u64).iter_be_bytes(f) }
}

impl int: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) { (self as uint).iter_le_bytes(f) }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) { (self as uint).iter_be_bytes(f) }
}

impl ~[const u8]: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) { f(self); }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) { f(self); }
}

impl @[const u8]: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) { f(self); }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) { f(self); }
}

impl<A: IterBytes> &[const A]: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) {
        for self.each |elt| {
            do elt.iter_le_bytes |bytes| {
                f(bytes)
            }
        }
    }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) {
        for self.each |elt| {
            do elt.iter_be_bytes |bytes| {
                f(bytes)
            }
        }
    }
}

fn iter_le_bytes_2<A: IterBytes, B: IterBytes>(a: &A, b: &B, f: Cb) {
    let mut flag = true;
    a.iter_le_bytes(|bytes| {flag = f(bytes); flag});
    if !flag { return; }
    b.iter_le_bytes(|bytes| {flag = f(bytes); flag});
}

fn iter_be_bytes_2<A: IterBytes, B: IterBytes>(a: &A, b: &B, f: Cb) {
    let mut flag = true;
    a.iter_be_bytes(|bytes| {flag = f(bytes); flag});
    if !flag { return; }
    b.iter_be_bytes(|bytes| {flag = f(bytes); flag});
}

fn iter_le_bytes_3<A: IterBytes,
                   B: IterBytes,
                   C: IterBytes>(a: &A, b: &B, c: &C, f: Cb) {
    let mut flag = true;
    a.iter_le_bytes(|bytes| {flag = f(bytes); flag});
    if !flag { return; }
    b.iter_le_bytes(|bytes| { flag = f(bytes); flag});
    if !flag { return; }
    c.iter_le_bytes(|bytes| {flag = f(bytes); flag});
}

fn iter_be_bytes_3<A: IterBytes,
                   B: IterBytes,
                   C: IterBytes>(a: &A, b: &B, c: &C, f: Cb) {
    let mut flag = true;
                       a.iter_be_bytes(|bytes| {flag = f(bytes); flag});
    if !flag { return; }
                       b.iter_be_bytes(|bytes| {flag = f(bytes); flag});
    if !flag { return; }
                       c.iter_be_bytes(|bytes| {flag = f(bytes); flag});
}

impl &str: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) {
        do str::byte_slice(self) |bytes| {
            f(bytes);
        }
    }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) {
        do str::byte_slice(self) |bytes| {
            f(bytes);
        }
    }
}

impl ~str: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) {
        do str::byte_slice(self) |bytes| {
            f(bytes);
        }
    }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) {
        do str::byte_slice(self) |bytes| {
            f(bytes);
        }
    }
}

impl @str: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) {
        do str::byte_slice(self) |bytes| {
            f(bytes);
        }
    }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) {
        do str::byte_slice(self) |bytes| {
            f(bytes);
        }
    }
}
impl<A: IterBytes> &A: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) {
        (*self).iter_le_bytes(f);
    }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) {
        (*self).iter_be_bytes(f);
    }
}

impl<A: IterBytes> @A: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) {
        (*self).iter_le_bytes(f);
    }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) {
        (*self).iter_be_bytes(f);
    }
}

impl<A: IterBytes> ~A: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) {
        (*self).iter_le_bytes(f);
    }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) {
        (*self).iter_be_bytes(f);
    }
}

// NB: raw-pointer IterBytes does _not_ dereference
// to the target; it just gives you the pointer-bytes.
impl<A> *A: IterBytes {
    #[inline(always)]
    fn iter_le_bytes(f: Cb) {
        (self as uint).iter_le_bytes(f);
    }
    #[inline(always)]
    fn iter_be_bytes(f: Cb) {
        (self as uint).iter_be_bytes(f);
    }
}


trait ToBytes {
    fn to_le_bytes() -> ~[u8];
    fn to_be_bytes() -> ~[u8];
}

impl<A: IterBytes> A: ToBytes {
    fn to_le_bytes() -> ~[u8] {
        let buf = io::mem_buffer();
        for self.iter_le_bytes |bytes| {
            buf.write(bytes)
        }
        io::mem_buffer_buf(buf)
    }
    fn to_be_bytes() -> ~[u8] {
        let buf = io::mem_buffer();
        for self.iter_be_bytes |bytes| {
            buf.write(bytes)
        }
        io::mem_buffer_buf(buf)
    }

}
