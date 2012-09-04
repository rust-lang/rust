struct contravariant {
    f: &int;
}

fn to_same_lifetime(bi: contravariant/&r) {
    let bj: contravariant/&r = bi;
}

fn to_shorter_lifetime(bi: contravariant/&r) {
    let bj: contravariant/&blk = bi;
}

fn to_longer_lifetime(bi: contravariant/&r) -> contravariant/&static {
    bi //~ ERROR mismatched types
}

fn main() {
}