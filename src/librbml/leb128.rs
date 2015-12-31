// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[inline]
pub fn write_to_vec(vec: &mut Vec<u8>, position: &mut usize, byte: u8)
{
    if *position == vec.len() {
        vec.push(byte);
    } else {
        vec[*position] = byte;
    }

    *position += 1;
}

pub fn write_unsigned_leb128(out: &mut Vec<u8>,
                             start_position: usize,
                             mut value: u64)
                             -> usize {
    let mut position = start_position;
    loop
    {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }

        write_to_vec(out, &mut position, byte);

        if value == 0 {
            break;
        }
    }

    return position - start_position;
}

pub fn read_unsigned_leb128(data: &[u8],
                            start_position: usize)
                            -> (u64, usize) {
    let mut result = 0;
    let mut shift = 0;
    let mut position = start_position;
    loop {
        let byte = data[position];
        position += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if (byte & 0x80) == 0 {
            break;
        }
        shift += 7;
    }

    (result, position - start_position)
}


pub fn write_signed_leb128(out: &mut Vec<u8>,
                           start_position: usize,
                           mut value: i64) -> usize {
    let mut position = start_position;

    loop {
        let mut byte = (value as u8) & 0x7f;
        value >>= 7;
        let more = !((((value == 0 ) && ((byte & 0x40) == 0)) ||
                      ((value == -1) && ((byte & 0x40) != 0))));
        if more {
            byte |= 0x80; // Mark this byte to show that more bytes will follow.
        }

        write_to_vec(out, &mut position, byte);

        if !more {
            break;
        }
    }

    return position - start_position;
}

pub fn read_signed_leb128(data: &[u8],
                          start_position: usize)
                          -> (i64, usize) {
    let mut result = 0;
    let mut shift = 0;
    let mut position = start_position;
    let mut byte;

    loop {
        byte = data[position];
        position += 1;
        result |= ((byte & 0x7F) as i64) << shift;
        shift += 7;

        if (byte & 0x80) == 0 {
            break;
        }
    }

    if (shift < 64) && ((byte & 0x40) != 0) {
        /* sign extend */
        result |= -(1i64 << shift);
    }

    (result, position - start_position)
}

#[test]
fn test_unsigned_leb128() {
    let mut stream = Vec::with_capacity(10000);

    for x in 0..62 {
        let pos = stream.len();
        let bytes_written = write_unsigned_leb128(&mut stream, pos, 3 << x);
        assert_eq!(stream.len(), pos + bytes_written);
    }

    let mut position = 0;
    for x in 0..62 {
        let expected = 3 << x;
        let (actual, bytes_read) = read_unsigned_leb128(&stream, position);
        assert_eq!(expected, actual);
        position += bytes_read;
    }
    assert_eq!(stream.len(), position);
}

#[test]
fn test_signed_leb128() {
    let mut values = Vec::new();

    let mut i = -500;
    while i < 500 {
        values.push(i * 123457i64);
        i += 1;
    }

    let mut stream = Vec::new();

    for &x in &values {
        let pos = stream.len();
        let bytes_written = write_signed_leb128(&mut stream, pos, x);
        assert_eq!(stream.len(), pos + bytes_written);
    }

    let mut pos = 0;
    for &x in &values {
        let (value, bytes_read) = read_signed_leb128(&mut stream, pos);
        pos += bytes_read;
        assert_eq!(x, value);
    }
    assert_eq!(pos, stream.len());
}
