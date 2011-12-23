use std;
import vec;

fn main() {
    let v =
        vec::map2([1, 2, 3, 4, 5],
                  [true, false, false, true, true],
                  {|i, b| if b { -i } else { i } });
    log(error, v);
    assert (v == [-1, 2, 3, -4, -5]);
}
