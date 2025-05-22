//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Fix for <https://github.com/rust-lang/rust/issues/125196>.

trait Tr {
    type Gat<T>;
}

struct W<T>(T);

fn foo<T: Tr>() where for<'a> &'a T: Tr<Gat<W<i32>> = i32> {
    let x: <&T as Tr>::Gat<W<_>> = 1i32;
    // Previously, `match_projection_projections` only checked that
    // `shallow_resolve(W<?0>) = W<?0>`. This won't prevent *all* inference guidance
    // from projection predicates in the environment, just ones that guide the
    // outermost type of each GAT constructor. This is definitely wrong, but there is
    // code that relies on it in the wild :/
}

fn main() {}
