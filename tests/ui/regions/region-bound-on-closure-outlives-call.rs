fn call_rec<F>(mut f: F) -> usize where F: FnMut(usize) -> usize {
    //~^ WARN function cannot return without recursing
    (|x| f(x))(call_rec(f)) //~ ERROR cannot move out of `f`
}

fn main() {}
