

// -*- rust -*-
extern mod std;

fn grow(v: &mut ~[int]) { *v += ~[1]; }

fn main() {
    let mut v: ~[int] = ~[];
    grow(&mut v);
    grow(&mut v);
    grow(&mut v);
    let len = vec::len::<int>(v);
    log(debug, len);
    assert (len == 3 as uint);
}
