import iter::iterable;
import str::chars_iter;
export iterable_by_chars;

impl iterable_by_chars of iterable<char> for str {
    fn iter(blk: fn(&&char)) {
        str::chars_iter(self, blk)
    }
}

#[test]
fn test_chars_iter() {
    let i = 0;
    iter::map("x\u03c0y") {|ch|
        alt i {
          0 { assert ch == 'x'; }
          1 { assert ch == '\u03c0'; }
          2 { assert ch == 'y'; }
        }
        i += 1;
    }

    iter::map("") {|_ch| fail; } // should not fail
}
