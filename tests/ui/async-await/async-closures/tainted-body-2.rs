//@ edition: 2021

// Ensure that building a by-ref async closure body doesn't ICE when the parent
// body is tainted.

fn main() {
    missing;
    //~^ ERROR cannot find value `missing` in this scope

    // We don't do numerical inference fallback when the body is tainted.
    // This leads to writeback folding the type of the coroutine-closure
    // into an error type, since its signature contains that numerical
    // infer var.
    let c = async |_| {};
    c(1);
}
