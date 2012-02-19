export iterable_by_bytes;
import iter::iterable;
import str::bytes_iter;

impl iterable_by_bytes of iterable<u8> for str {
    fn iter(blk: fn(&&u8)) {
        str::bytes_iter(self, blk)
    }
}

#[test]
fn test_bytes_iter() {
    let i = 0;

    iter::map("xyz") {|bb|
        alt i {
          0 { assert bb == 'x' as u8; }
          1 { assert bb == 'y' as u8; }
          2 { assert bb == 'z' as u8; }
        }
        i += 1;
    }

    iter::map("") {|bb| assert bb == 0u8; }
}

