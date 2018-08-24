struct invariant<'a> {
    f: Box<FnOnce(&mut &'a isize) + 'static>,
}

fn to_same_lifetime<'r>(b_isize: invariant<'r>) {
    let bj: invariant<'r> = b_isize;
}

fn to_longer_lifetime<'r>(b_isize: invariant<'r>) -> invariant<'static> {
    b_isize //~ ERROR mismatched types
}

fn main() {
}
