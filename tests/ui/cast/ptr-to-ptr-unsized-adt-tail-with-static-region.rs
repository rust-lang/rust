//@ check-pass

// During MIR typeck when casting `*mut dyn Sync + '?x` to
// `*mut Wrap` we compute the tail of `Wrap` as `dyn Sync + 'static`.
//
// This test ensures that we first convert the `'static` lifetime to
// the nll var `'?0` before introducing the region constraint `'?x: 'static`.

struct Wrap(dyn Sync + 'static);

fn cast(x: *mut (dyn Sync + 'static)) {
    x as *mut Wrap;
}

fn main() {}
