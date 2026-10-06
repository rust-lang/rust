//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #135617

// The bound `for<'a> (): Project` prevents us from normalizing the
// array length. This caused an ICE with the new solver. The actual
// behavior doesn't really matter here as long as it doesn't ICE.

trait Project {
    const ASSOC: usize;
}

fn foo()
where
    for<'a> (): Project,
{
    [(); <() as Project>::ASSOC];
    //[next]~^ ERROR: type annotations needed
}

pub fn main() {}
