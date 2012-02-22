import iter::iterable;
export iterable_by_chars;

impl iterable_by_chars of iterable<char> for str {
    fn iter(blk: fn(&&char)) {
        str::chars_iter(self) { |ch| blk(ch) }
    }
}

#[test]
fn test_str_char_iter() {
    let i = 0u;
    "፩፪፫".iter {|&&c: char|
        alt i {
          0u { assert '፩' == c }
          1u { assert '፪' == c }
          2u { assert '፫' == c }
          _ { fail; }
        }
        i += 1u;
    }
}
