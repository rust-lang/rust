// Contravariant with respect to a region:
//
// You can upcast to a *smaller region* but not a larger one.  This is
// the normal case.

struct contravariant {
    f: fn@() -> &self/int;
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