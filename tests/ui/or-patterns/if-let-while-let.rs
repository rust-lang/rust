// Check that or patterns are lowered correctly in `if let` and `while let` expressions.
//@ run-pass

fn main() {
    let mut opt = Some(3);
    let mut w = Vec::new();
    while let Some(ref mut val @ (3 | 4 | 6)) = opt {
        w.push(*val);
        *val += 1;
    }
    assert_eq!(w, [3, 4]);
    if let &(None | Some(6 | 7)) = &opt {
        unreachable!();
    }
    if let Some(x @ (4 | 5 | 6)) = opt {
        assert_eq!(x, 5);
    } else {
        unreachable!();
    }
}
