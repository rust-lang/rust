pub(crate) fn write_signed_vlqhex_to_string(n: i32, string: &mut String) {
    let (sign, magnitude): (bool, u32) =
        if n >= 0 { (false, n.try_into().unwrap()) } else { (true, (-n).try_into().unwrap()) };
    // zig-zag encoding
    let value: u32 = (magnitude << 1) | (if sign { 1 } else { 0 });
    // Self-terminating hex use capital letters for everything but the
    // least significant digit, which is lowercase. For example, decimal 17
    // would be `` Aa `` if zig-zag encoding weren't used.
    //
    // Zig-zag encoding, however, stores the sign bit as the last bit.
    // This means, in the last hexit, 1 is actually `c`, -1 is `b`
    // (`a` is the imaginary -0), and, because all the bits are shifted
    // by one, `` A` `` is actually 8 and `` Aa `` is -8.
    //
    // https://rust-lang.github.io/rustc-dev-guide/rustdoc-internals/search.html
    // describes the encoding in more detail.
    let mut shift: u32 = 28;
    let mut mask: u32 = 0xF0_00_00_00;
    // first skip leading zeroes
    while shift < 32 {
        let hexit = (value & mask) >> shift;
        if hexit != 0 || shift == 0 {
            break;
        }
        shift = shift.wrapping_sub(4);
        mask >>= 4;
    }
    // now write the rest
    while shift < 32 {
        let hexit = (value & mask) >> shift;
        let hex = char::try_from(if shift == 0 { '`' } else { '@' } as u32 + hexit).unwrap();
        string.push(hex);
        shift = shift.wrapping_sub(4);
        mask >>= 4;
    }
}

pub fn read_signed_vlqhex_from_string(string: &[u8]) -> Option<(i32, usize)> {
    let mut n = 0i32;
    let mut i = 0;
    while let Some(&c) = string.get(i) {
        i += 1;
        n = (n << 4) | i32::from(c & 0xF);
        if c >= 96 {
            // zig-zag encoding
            let (sign, magnitude) = (n & 1, n >> 1);
            let value = if sign == 0 { 1 } else { -1 } * magnitude;
            return Some((value, i));
        }
    }
    None
}

pub fn write_postings_to_string(postings: &[Vec<u32>], buf: &mut Vec<u8>) {
    for list in postings {
        if list.is_empty() {
            buf.push(0);
            continue;
        }
        let len_before = buf.len();
        stringdex::internals::encode::write_bitmap_to_bytes(&list, &mut *buf).unwrap();
        let len_after = buf.len();
        if len_after - len_before > 1 + (4 * list.len()) && list.len() < 0x3a {
            buf.truncate(len_before);
            buf.push(list.len() as u8);
            for &item in list {
                buf.push(item as u8);
                buf.push((item >> 8) as u8);
                buf.push((item >> 16) as u8);
                buf.push((item >> 24) as u8);
            }
        }
    }
}

pub fn read_postings_from_string(postings: &mut Vec<Vec<u32>>, mut buf: &[u8]) {
    use stringdex::internals::decode::RoaringBitmap;
    while let Some(&c) = buf.get(0) {
        if c < 0x3a {
            buf = &buf[1..];
            let mut slot = Vec::new();
            for _ in 0..c {
                slot.push(
                    (buf[0] as u32)
                        | ((buf[1] as u32) << 8)
                        | ((buf[2] as u32) << 16)
                        | ((buf[3] as u32) << 24),
                );
                buf = &buf[4..];
            }
            postings.push(slot);
        } else {
            let (bitmap, consumed_bytes_len) =
                RoaringBitmap::from_bytes(buf).unwrap_or_else(|| (RoaringBitmap::default(), 0));
            assert_ne!(consumed_bytes_len, 0);
            postings.push(bitmap.to_vec());
            buf = &buf[consumed_bytes_len..];
        }
    }
}
