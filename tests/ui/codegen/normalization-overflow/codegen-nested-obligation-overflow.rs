//@ revisions: old next
//[next]~^ ERROR overflow evaluating the requirement
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver=globally -Arecursion_depth_exceeding_limit
//@ build-fail
//@ dont-check-compiler-stderr

// Nested obligation overflow during codegen selection must produce
// a compiler error instead of an ICE while resolving Iterator::peekable.

fn main() {
    let mut items = vec![1, 2, 3, 4, 5].into_iter();
    recurse(&mut items);
}

fn recurse(items: &mut impl Iterator<Item = u8>) {
    let mut peeker = items.peekable();
    //[old]~^ ERROR reached the recursion limit while instantiating

    if peeker.peek().is_none() {
        return;
    }

    recurse(&mut peeker);
}
