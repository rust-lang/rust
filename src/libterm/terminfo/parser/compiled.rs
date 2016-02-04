// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! ncurses-compatible compiled terminfo format parsing (term(5))

use std::collections::HashMap;
use std::io::prelude::*;
use std::io;

use terminfo::Error::*;
use terminfo::TermInfo;
use Result;

pub use terminfo::parser::names::*;

// These are the orders ncurses uses in its compiled format (as of 5.9). Not
// sure if portable.

fn read_le_u16(r: &mut io::Read) -> io::Result<u16> {
    let mut b = [0; 2];
    let mut amt = 0;
    while amt < b.len() {
        match try!(r.read(&mut b[amt..])) {
            0 => return Err(io::Error::new(io::ErrorKind::Other, "end of file")),
            n => amt += n,
        }
    }
    Ok((b[0] as u16) | ((b[1] as u16) << 8))
}

fn read_byte(r: &mut io::Read) -> io::Result<u8> {
    match r.bytes().next() {
        Some(s) => s,
        None => Err(io::Error::new(io::ErrorKind::Other, "end of file")),
    }
}

/// Parse a compiled terminfo entry, using long capability names if `longnames`
/// is true
pub fn parse(file: &mut io::Read, longnames: bool) -> Result<TermInfo> {
    let (bnames, snames, nnames) = if longnames {
        (boolfnames, stringfnames, numfnames)
    } else {
        (boolnames, stringnames, numnames)
    };

    // Check magic number
    let magic = try!(read_le_u16(file));
    if magic != 0x011A {
        return Err(BadMagic(magic).into());
    }

    // According to the spec, these fields must be >= -1 where -1 means that the
    // feature is not
    // supported. Using 0 instead of -1 works because we skip sections with length
    // 0.
    macro_rules! read_nonneg {
        () => {{
            match try!(read_le_u16(file)) as i16 {
                n if n >= 0 => n as usize,
                -1 => 0,
                _ => return Err(InvalidLength.into()),
            }
        }}
    }

    let names_bytes = read_nonneg!();
    let bools_bytes = read_nonneg!();
    let numbers_count = read_nonneg!();
    let string_offsets_count = read_nonneg!();
    let string_table_bytes = read_nonneg!();

    if names_bytes == 0 {
        return Err(ShortNames.into());
    }

    if bools_bytes > boolnames.len() {
        return Err(TooManyBools.into());
    }

    if numbers_count > numnames.len() {
        return Err(TooManyNumbers.into());
    }

    if string_offsets_count > stringnames.len() {
        return Err(TooManyStrings.into());
    }

    // don't read NUL
    let mut bytes = Vec::new();
    try!(file.take((names_bytes - 1) as u64).read_to_end(&mut bytes));
    let names_str = match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => return Err(NotUtf8(e.utf8_error()).into()),
    };

    let term_names: Vec<String> = names_str.split('|')
                                           .map(|s| s.to_string())
                                           .collect();
    // consume NUL
    if try!(read_byte(file)) != b'\0' {
        return Err(NamesMissingNull.into());
    }

    let bools_map: HashMap<String, bool> = try! {
        (0..bools_bytes).filter_map(|i| match read_byte(file) {
            Err(e) => Some(Err(e)),
            Ok(1) => Some(Ok((bnames[i].to_string(), true))),
            Ok(_) => None
        }).collect()
    };

    if (bools_bytes + names_bytes) % 2 == 1 {
        try!(read_byte(file)); // compensate for padding
    }

    let numbers_map: HashMap<String, u16> = try! {
        (0..numbers_count).filter_map(|i| match read_le_u16(file) {
            Ok(0xFFFF) => None,
            Ok(n) => Some(Ok((nnames[i].to_string(), n))),
            Err(e) => Some(Err(e))
        }).collect()
    };

    let string_map: HashMap<String, Vec<u8>> = if string_offsets_count > 0 {
        let string_offsets: Vec<u16> = try!((0..string_offsets_count)
                                                .map(|_| read_le_u16(file))
                                                .collect());

        let mut string_table = Vec::new();
        try!(file.take(string_table_bytes as u64).read_to_end(&mut string_table));

        try!(string_offsets.into_iter()
                           .enumerate()
                           .filter(|&(_, offset)| {
                               // non-entry
                               offset != 0xFFFF
                           })
                           .map(|(i, offset)| {
                               let offset = offset as usize;

                               let name = if snames[i] == "_" {
                                   stringfnames[i]
                               } else {
                                   snames[i]
                               };

                               if offset == 0xFFFE {
                                   // undocumented: FFFE indicates cap@, which means the capability
                                   // is not present
                                   // unsure if the handling for this is correct
                                   return Ok((name.to_string(), Vec::new()));
                               }

                               // Find the offset of the NUL we want to go to
                               let nulpos = string_table[offset..string_table_bytes]
                                                .iter()
                                                .position(|&b| b == 0);
                               match nulpos {
                                   Some(len) => {
                                       Ok((name.to_string(),
                                           string_table[offset..offset + len].to_vec()))
                                   }
                                   None => return Err(::Error::TerminfoParsing(StringsMissingNull)),
                               }
                           })
                           .collect())
    } else {
        HashMap::new()
    };

    // And that's all there is to it
    Ok(TermInfo {
        names: term_names,
        bools: bools_map,
        numbers: numbers_map,
        strings: string_map,
    })
}

/// Create a dummy TermInfo struct for msys terminals
pub fn msys_terminfo() -> TermInfo {
    let mut strings = HashMap::new();
    strings.insert("sgr0".to_string(), b"\x1B[0m".to_vec());
    strings.insert("bold".to_string(), b"\x1B[1m".to_vec());
    strings.insert("setaf".to_string(), b"\x1B[3%p1%dm".to_vec());
    strings.insert("setab".to_string(), b"\x1B[4%p1%dm".to_vec());

    let mut numbers = HashMap::new();
    numbers.insert("colors".to_string(), 8u16);

    TermInfo {
        names: vec!["cygwin".to_string()], // msys is a fork of an older cygwin version
        bools: HashMap::new(),
        numbers: numbers,
        strings: strings,
    }
}

#[cfg(test)]
mod test {

    use super::{boolnames, boolfnames, numnames, numfnames, stringnames, stringfnames};

    #[test]
    fn test_veclens() {
        assert_eq!(boolfnames.len(), boolnames.len());
        assert_eq!(numfnames.len(), numnames.len());
        assert_eq!(stringfnames.len(), stringnames.len());
    }
}
