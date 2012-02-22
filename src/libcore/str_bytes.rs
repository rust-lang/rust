import iter::iterable;
export iterable_by_bytes;

impl iterable_by_bytes of iterable<u8> for str {
    fn iter(blk: fn(&&u8)) {
        str::bytes_iter(self) { |b| blk(b) }
    }
}

#[test]
fn test_str_byte_iter() {
    let i = 0u;
    "፩፪፫".iter {|&&b: u8|
        alt i {
          0u { assert 0xe1 as u8 == b }
          1u { assert 0x8d as u8 == b }
          2u { assert 0xa9 as u8 == b }
          4u { assert 0xe1 as u8 == b }
          5u { assert 0x8d as u8 == b }
          6u { assert 0xaa as u8 == b }
          7u { assert 0xe1 as u8 == b }
          8u { assert 0x8d as u8 == b }
          9u { assert 0xab as u8 == b }
          _ { fail; }
        }
        i += 1u;
    }
}
