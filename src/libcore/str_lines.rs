export iterable_by_lines;
import iter::iterable;
import str::lines_iter;

impl iterable_by_lines of iterable<str> for str {
    fn iter(blk: fn(&&str)) {
        str::lines_iter(self, blk)
    }
}

#[test]
fn test_lines_iter () {
    let lf = "\nMary had a little lamb\nLittle lamb\n";

    let ii = 0;

    iter::map(lf) {|x|
        alt ii {
            0 { assert "" == x; }
            1 { assert "Mary had a little lamb" == x; }
            2 { assert "Little lamb" == x; }
            3 { assert "" == x; }
            _ { () }
        }
        ii += 1;
    }
}
