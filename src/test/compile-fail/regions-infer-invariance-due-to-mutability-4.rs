struct invariant {
    f: fn@() -> @mut &self/int;
}

fn to_same_lifetime(bi: invariant/&r) {
    let bj: invariant/&r = bi;
}

fn to_shorter_lifetime(bi: invariant/&r) {
    let bj: invariant/&blk = bi; //~ ERROR mismatched types
}

fn to_longer_lifetime(bi: invariant/&r) -> invariant/&static {
    bi //~ ERROR mismatched types
}

fn main() {
}