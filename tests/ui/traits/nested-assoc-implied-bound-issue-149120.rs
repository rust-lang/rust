//@ revisions: current next
//@ [next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/149120>.
// Used to result in delayed bug

fn f<T: WithAssoc<Assoc = ()>>(_ptr: &Wrapper<T>) {}

struct Wrapper<T: WithAssoc>(<T::Assoc as WithAssoc>::Assoc);

trait WithAssoc {
    type Assoc: WithAssoc;
}

fn main() {}
