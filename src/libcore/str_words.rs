export iterable_by_words;
import iter::iterable;
import str::words_iter;

impl iterable_by_words of iterable<str> for str {
    fn iter(blk: fn(&&str)) {
        str::words_iter(self, blk)
    }
}

#[test]
fn test_words_iter() {
    let data = "\nMary had a little lamb\nLittle lamb\n";

    let ii = 0;

    iter::map(data) {|ww|
        alt ii {
          0 { assert "Mary"   == ww; }
          1 { assert "had"    == ww; }
          2 { assert "a"      == ww; }
          3 { assert "little" == ww; }
          _ { () }
        }
        ii += 1;
    }

    iter::map("") {|_x| fail; } // should not fail
}
