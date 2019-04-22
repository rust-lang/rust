// compile-pass

union Uninit {
    _never_use: *const u8,
    uninit: (),
}

const UNINIT: Uninit = Uninit { uninit: () };

fn main() {}
