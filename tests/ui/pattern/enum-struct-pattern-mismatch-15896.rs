//! Regression test for https://github.com/rust-lang/rust/issues/15896

// Regression test for #15896. It used to ICE rustc.

fn main() {
    enum R { REB(()) }
    struct Tau { t: usize }
    enum E { B(R, Tau) }

    let e = E::B(R::REB(()), Tau { t: 3 });
    let u = match e {
        E::B(
          Tau{t: x},
          //~^ ERROR mismatched types
          _) => x,
    };
}
