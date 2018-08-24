fn call_rec<F>(mut f: F) -> usize where F: FnMut(usize) -> usize {
    (|x| f(x))(call_rec(f)) //~ ERROR cannot move out of `f`
}

fn main() {}
