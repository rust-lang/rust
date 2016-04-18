// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use session::Session;

use std::cmp;

/// Parsed [Data layout](http://llvm.org/docs/LangRef.html#data-layout)
/// for a target, which contains everything needed to compute layouts.
pub struct TargetDataLayout {
    pub endian: Endian,
    pub i1_align: Align,
    pub i8_align: Align,
    pub i16_align: Align,
    pub i32_align: Align,
    pub i64_align: Align,
    pub f32_align: Align,
    pub f64_align: Align,
    pub pointer_size: Size,
    pub pointer_align: Align,
    pub aggregate_align: Align,

    /// Alignments for vector types, sorted by size.
    pub vector_align: Vec<(Size, Align)>
}

impl Default for TargetDataLayout {
    fn default() -> TargetDataLayout {
        TargetDataLayout {
            endian: Endian::Big,
            i1_align: Align::from_bits(8, 8).unwrap(),
            i8_align: Align::from_bits(8, 8).unwrap(),
            i16_align: Align::from_bits(16, 16).unwrap(),
            i32_align: Align::from_bits(32, 32).unwrap(),
            i64_align: Align::from_bits(32, 64).unwrap(),
            f32_align: Align::from_bits(32, 32).unwrap(),
            f64_align: Align::from_bits(64, 64).unwrap(),
            pointer_size: Size::from_bits(64),
            pointer_align: Align::from_bits(64, 64).unwrap(),
            aggregate_align: Align::from_bits(0, 64).unwrap(),
            vector_align: vec![(Size::from_bits(128),
                                Align::from_bits(128, 128).unwrap())]
        }
    }
}

impl TargetDataLayout {
    pub fn parse(sess: &Session) -> TargetDataLayout {
        // Parse a bit count from a string.
        let parse_bits = |s: &str, kind: &str, cause: &str| {
            s.parse::<u64>().unwrap_or_else(|err| {
                sess.err(&format!("invalid {} `{}` for `{}` in \"data-layout\": {}",
                                  kind, s, cause, err));
                0
            })
        };

        // Parse a size string.
        let size = |s: &str, cause: &str| {
            Size::from_bits(parse_bits(s, "size", cause))
        };

        // Parse an alignment string.
        let align = |s: &[&str], cause: &str| {
            if s.is_empty() {
                sess.err(&format!("missing alignment for `{}` in \"data-layout\"", cause));
            }
            let abi = parse_bits(s[0], "alignment", cause);
            let pref = s.get(1).map_or(abi, |pref| parse_bits(pref, "alignment", cause));
            Align::from_bits(abi, pref).unwrap_or_else(|err| {
                sess.err(&format!("invalid alignment for `{}` in \"data-layout\": {}",
                                  cause, err));
                Align::from_bits(8, 8).unwrap()
            })
        };

        let mut dl = TargetDataLayout::default();
        for spec in sess.target.target.data_layout.split("-") {
            match &spec.split(":").collect::<Vec<_>>()[..] {
                ["e"] => dl.endian = Endian::Little,
                ["E"] => dl.endian = Endian::Big,
                ["a", a..] => dl.aggregate_align = align(a, "a"),
                ["f32", a..] => dl.f32_align = align(a, "f32"),
                ["f64", a..] => dl.f64_align = align(a, "f64"),
                [p @ "p", s, a..] | [p @ "p0", s, a..] => {
                    dl.pointer_size = size(s, p);
                    dl.pointer_align = align(a, p);
                }
                [s, a..] if s.starts_with("i") => {
                    let ty_align = match s[1..].parse::<u64>() {
                        Ok(1) => &mut dl.i8_align,
                        Ok(8) => &mut dl.i8_align,
                        Ok(16) => &mut dl.i16_align,
                        Ok(32) => &mut dl.i32_align,
                        Ok(64) => &mut dl.i64_align,
                        Ok(_) => continue,
                        Err(_) => {
                            size(&s[1..], "i"); // For the user error.
                            continue;
                        }
                    };
                    *ty_align = align(a, s);
                }
                [s, a..] if s.starts_with("v") => {
                    let v_size = size(&s[1..], "v");
                    let a = align(a, s);
                    if let Some(v) = dl.vector_align.iter_mut().find(|v| v.0 == v_size) {
                        v.1 = a;
                        continue;
                    }
                    // No existing entry, add a new one.
                    dl.vector_align.push((v_size, a));
                }
                _ => {} // Ignore everything else.
            }
        }

        // Sort vector alignments by size.
        dl.vector_align.sort_by_key(|&(s, _)| s);

        // Perform consistency checks against the Target information.
        let endian_str = match dl.endian {
            Endian::Little => "little",
            Endian::Big => "big"
        };
        if endian_str != sess.target.target.target_endian {
            sess.err(&format!("inconsistent target specification: \"data-layout\" claims \
                               architecture is {}-endian, while \"target-endian\" is `{}`",
                              endian_str, sess.target.target.target_endian));
        }

        if dl.pointer_size.bits().to_string() != sess.target.target.target_pointer_width {
            sess.err(&format!("inconsistent target specification: \"data-layout\" claims \
                               pointers are {}-bit, while \"target-pointer-width\" is `{}`",
                              dl.pointer_size.bits(), sess.target.target.target_pointer_width));
        }

        dl
    }
}

/// Endianness of the target, which must match cfg(target-endian).
#[derive(Copy, Clone)]
pub enum Endian {
    Little,
    Big
}

/// Size of a type in bytes.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Size {
    raw: u64
}

impl Size {
    pub fn from_bits(bits: u64) -> Size {
        Size::from_bytes((bits + 7) / 8)
    }

    pub fn from_bytes(bytes: u64) -> Size {
        if bytes >= (1 << 61) {
            bug!("Size::from_bytes: {} bytes in bits doesn't fit in u64", bytes)
        }
        Size {
            raw: bytes
        }
    }

    pub fn bytes(self) -> u64 {
        self.raw
    }

    pub fn bits(self) -> u64 {
        self.bytes() * 8
    }
}

/// Alignment of a type in bytes, both ABI-mandated and preferred.
/// Since alignments are always powers of 2, we can pack both in one byte,
/// giving each a nibble (4 bits) for a maximum alignment of 2^15 = 32768.
#[derive(Copy, Clone)]
pub struct Align {
    raw: u8
}

impl Align {
    pub fn from_bits(abi: u64, pref: u64) -> Result<Align, String> {
        Align::from_bytes((abi + 7) / 8, (pref + 7) / 8)
    }

    pub fn from_bytes(abi: u64, pref: u64) -> Result<Align, String> {
        let pack = |align: u64| {
            // Treat an alignment of 0 bytes like 1-byte alignment.
            if align == 0 {
                return Ok(0);
            }

            let mut bytes = align;
            let mut pow: u8 = 0;
            while (bytes & 1) == 0 {
                pow += 1;
                bytes >>= 1;
            }
            if bytes != 1 {
                Err(format!("`{}` is not a power of 2", align))
            } else if pow > 0x0f {
                Err(format!("`{}` is too large", align))
            } else {
                Ok(pow)
            }
        };

        Ok(Align {
            raw: pack(abi)? | (pack(pref)? << 4)
        })
    }

    pub fn abi(self) -> u64 {
        1 << (self.raw & 0xf)
    }

    pub fn pref(self) -> u64 {
        1 << (self.raw >> 4)
    }

    pub fn min(self, other: Align) -> Align {
        let abi = cmp::min(self.raw & 0x0f, other.raw & 0x0f);
        let pref = cmp::min(self.raw & 0xf0, other.raw & 0xf0);
        Align {
            raw: abi | pref
        }
    }

    pub fn max(self, other: Align) -> Align {
        let abi = cmp::max(self.raw & 0x0f, other.raw & 0x0f);
        let pref = cmp::max(self.raw & 0xf0, other.raw & 0xf0);
        Align {
            raw: abi | pref
        }
    }
}
