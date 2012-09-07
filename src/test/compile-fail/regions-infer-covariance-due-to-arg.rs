// Covariant with respect to a region:
//
// You can upcast to a *larger region* but not a smaller one.

struct covariant {
    f: fn@(x: &self/int) -> int
}

fn to_same_lifetime(bi: covariant/&r) {
    let bj: covariant/&r = bi;
}

fn to_shorter_lifetime(bi: covariant/&r) {
    let bj: covariant/&blk = bi; //~ ERROR mismatched types
}

fn to_longer_lifetime(bi: covariant/&r) -> covariant/&static {
    bi
}

fn main() {
}