type rec = {
    f: int
};

fn destructure(x: &mut rec) {
    alt *x {
      {f: ref mut f} => *f += 1
    }
}

fn main() {
    let mut v = {f: 22};
    destructure(&mut v);
    assert v.f == 23;
}