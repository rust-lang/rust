//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver=globally -Arecursion_depth_exceeding_limit
//@ build-fail
//@ dont-check-compiler-stderr
//@ forbid-output: internal compiler error

// Recursive Peekable instantiation can overflow during codegen.
// The next solver should report an error instead of ICEing.
//[next]~? ERROR overflow evaluating the requirement

fn main() {
    let mut items = vec![1, 2, 3, 4, 5].into_iter();
    problem_thingy(&mut items);
}

fn problem_thingy(items: &mut impl Iterator<Item = u8>) {
    let mut peeker = items.peekable();
    //[current]~^ ERROR: reached the recursion limit while instantiating
    match peeker.peek() {
        Some(_) => (),
        None => return (),
    }
    problem_thingy(&mut peeker);
}
