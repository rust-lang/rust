//@ build-pass (FIXME(62277): could be check-pass?)

union Uninit {
    _never_use: *const u8,
    uninit: (),
}

const UNINIT: Uninit = Uninit { uninit: () };
const UNINIT2: (Uninit,) = (Uninit { uninit: () }, );

fn main() {}
