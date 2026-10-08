//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #127033
//@ edition: 2021

// Regression test for #127033. In the old solver this test only constrained
// the RPIT in dead code, causing MIR typeck to not define it at all.
//
// This then caused us to use the type inferred by HIR typeck, later resulting in
// ICE when leaking `'erased` to parts of the compiler which didn't expect it.
//
// cc trait-system-refactor-initiative#170

pub trait RaftLogStorage {
    fn save_vote(vote: ()) -> impl std::future::Future + Send;
}

struct X;
impl RaftLogStorage for X {
    fn save_vote(vote: ()) -> impl std::future::Future {
        loop {}
        async {
            &vote
        }
    }
}

fn main() {}
