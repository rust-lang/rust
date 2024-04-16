//@ known-bug: #121299
#[derive(Eq)]
struct D {
    _: union {
    },
}
