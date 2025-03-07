use base64::prelude::*;

pub(crate) fn write_vlqhex_to_string(n: i32, string: &mut String) {
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

// Used during bitmap encoding
enum Container {
    /// number of ones, bits
    Bits(Box<[u64; 1024]>),
    /// list of entries
    Array(Vec<u16>),
    /// list of (start, len-1)
    Run(Vec<(u16, u16)>),
}
impl Container {
    fn popcount(&self) -> u32 {
        match self {
            Container::Bits(bits) => bits.iter().copied().map(|x| x.count_ones()).sum(),
            Container::Array(array) => {
                array.len().try_into().expect("array can't be bigger than 2**32")
            }
            Container::Run(runs) => {
                runs.iter().copied().map(|(_, lenm1)| u32::from(lenm1) + 1).sum()
            }
        }
    }
    fn push(&mut self, value: u16) {
        match self {
            Container::Bits(bits) => bits[value as usize >> 6] |= 1 << (value & 0x3F),
            Container::Array(array) => {
                array.push(value);
                if array.len() >= 4096 {
                    let array = std::mem::take(array);
                    *self = Container::Bits(Box::new([0; 1024]));
                    for value in array {
                        self.push(value);
                    }
                }
            }
            Container::Run(runs) => {
                if let Some(r) = runs.last_mut()
                    && r.0 + r.1 + 1 == value
                {
                    r.1 += 1;
                } else {
                    runs.push((value, 0));
                }
            }
        }
    }
    fn try_make_run(&mut self) -> bool {
        match self {
            Container::Bits(bits) => {
                let mut r: u64 = 0;
                for (i, chunk) in bits.iter().copied().enumerate() {
                    let next_chunk =
                        i.checked_add(1).and_then(|i| bits.get(i)).copied().unwrap_or(0);
                    r += !chunk & u64::from((chunk << 1).count_ones());
                    r += !next_chunk & u64::from((chunk >> 63).count_ones());
                }
                if (2 + 4 * r) >= 8192 {
                    return false;
                }
                let bits = std::mem::replace(bits, Box::new([0; 1024]));
                *self = Container::Run(Vec::new());
                for (i, bits) in bits.iter().copied().enumerate() {
                    if bits == 0 {
                        continue;
                    }
                    for j in 0..64 {
                        let value = (u16::try_from(i).unwrap() << 6) | j;
                        if bits & (1 << j) != 0 {
                            self.push(value);
                        }
                    }
                }
                true
            }
            Container::Array(array) if array.len() <= 5 => false,
            Container::Array(array) => {
                let mut r = 0;
                let mut prev = None;
                for value in array.iter().copied() {
                    if value.checked_sub(1) != prev {
                        r += 1;
                    }
                    prev = Some(value);
                }
                if 2 + 4 * r >= 2 * array.len() + 2 {
                    return false;
                }
                let array = std::mem::take(array);
                *self = Container::Run(Vec::new());
                for value in array {
                    self.push(value);
                }
                true
            }
            Container::Run(_) => true,
        }
    }
}

// checked against roaring-rs in
// https://gitlab.com/notriddle/roaring-test
pub(crate) fn write_bitmap_to_bytes(
    domain: &[u32],
    mut out: impl std::io::Write,
) -> std::io::Result<()> {
    // https://arxiv.org/pdf/1603.06549.pdf
    let mut keys = Vec::<u16>::new();
    let mut containers = Vec::<Container>::new();
    let mut key: u16;
    let mut domain_iter = domain.iter().copied().peekable();
    let mut has_run = false;
    while let Some(entry) = domain_iter.next() {
        key = (entry >> 16).try_into().expect("shifted off the top 16 bits, so it should fit");
        let value: u16 = (entry & 0x00_00_FF_FF).try_into().expect("AND 16 bits, so it should fit");
        let mut container = Container::Array(vec![value]);
        while let Some(entry) = domain_iter.peek().copied() {
            let entry_key: u16 =
                (entry >> 16).try_into().expect("shifted off the top 16 bits, so it should fit");
            if entry_key != key {
                break;
            }
            domain_iter.next().expect("peeking just succeeded");
            container
                .push((entry & 0x00_00_FF_FF).try_into().expect("AND 16 bits, so it should fit"));
        }
        keys.push(key);
        has_run = container.try_make_run() || has_run;
        containers.push(container);
    }
    // https://github.com/RoaringBitmap/RoaringFormatSpec
    const SERIAL_COOKIE_NO_RUNCONTAINER: u32 = 12346;
    const SERIAL_COOKIE: u32 = 12347;
    const NO_OFFSET_THRESHOLD: u32 = 4;
    let size: u32 = containers.len().try_into().unwrap();
    let start_offset = if has_run {
        out.write_all(&u32::to_le_bytes(SERIAL_COOKIE | ((size - 1) << 16)))?;
        for set in containers.chunks(8) {
            let mut b = 0;
            for (i, container) in set.iter().enumerate() {
                if matches!(container, &Container::Run(..)) {
                    b |= 1 << i;
                }
            }
            out.write_all(&[b])?;
        }
        if size < NO_OFFSET_THRESHOLD {
            4 + 4 * size + size.div_ceil(8)
        } else {
            4 + 8 * size + size.div_ceil(8)
        }
    } else {
        out.write_all(&u32::to_le_bytes(SERIAL_COOKIE_NO_RUNCONTAINER))?;
        out.write_all(&u32::to_le_bytes(containers.len().try_into().unwrap()))?;
        4 + 4 + 4 * size + 4 * size
    };
    for (&key, container) in keys.iter().zip(&containers) {
        // descriptive header
        let key: u32 = key.into();
        let count: u32 = container.popcount() - 1;
        out.write_all(&u32::to_le_bytes((count << 16) | key))?;
    }
    if !has_run || size >= NO_OFFSET_THRESHOLD {
        // offset header
        let mut starting_offset = start_offset;
        for container in &containers {
            out.write_all(&u32::to_le_bytes(starting_offset))?;
            starting_offset += match container {
                Container::Bits(_) => 8192u32,
                Container::Array(array) => u32::try_from(array.len()).unwrap() * 2,
                Container::Run(runs) => 2 + u32::try_from(runs.len()).unwrap() * 4,
            };
        }
    }
    for container in &containers {
        match container {
            Container::Bits(bits) => {
                for chunk in bits.iter() {
                    out.write_all(&u64::to_le_bytes(*chunk))?;
                }
            }
            Container::Array(array) => {
                for value in array.iter() {
                    out.write_all(&u16::to_le_bytes(*value))?;
                }
            }
            Container::Run(runs) => {
                out.write_all(&u16::to_le_bytes(runs.len().try_into().unwrap()))?;
                for (start, lenm1) in runs.iter().copied() {
                    out.write_all(&u16::to_le_bytes(start))?;
                    out.write_all(&u16::to_le_bytes(lenm1))?;
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn bitmap_to_string(domain: &[u32]) -> String {
    let mut buf = Vec::new();
    let mut strbuf = String::new();
    write_bitmap_to_bytes(domain, &mut buf).unwrap();
    BASE64_STANDARD.encode_string(&buf, &mut strbuf);
    strbuf
}
