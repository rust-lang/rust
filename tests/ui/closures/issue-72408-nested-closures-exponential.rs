// build-pass
// ignore-compare-mode-next-solver (hangs)

// Closures include captured types twice in a type tree.
//
// Wrapping one closure with another leads to doubling
// the amount of types in the type tree.
//
// This test ensures that rust can handle
// deeply nested type trees with a lot
// of duplicated subtrees.

fn dup(f: impl Fn(i32) -> i32) -> impl Fn(i32) -> i32 {
    move |a| f(a * 2)
}

fn main() {
    let f = |a| a;

    let f = dup(f);
    let f = dup(f);
    let f = dup(f);
    let f = dup(f);
    let f = dup(f);

    let f = dup(f);
    let f = dup(f);
    let f = dup(f);
    let f = dup(f);
    let f = dup(f);

    let f = dup(f);
    let f = dup(f);
    let f = dup(f);
    let f = dup(f);
    let f = dup(f);

    let f = dup(f);
    let f = dup(f);
    let f = dup(f);
    let f = dup(f);
    let f = dup(f);

    // Compiler dies around here if it tries
    // to walk the tree exhaustively.

    let f = dup(f);
    let f = dup(f);
    let f = dup(f);
    let f = dup(f);
    let f = dup(f);

    let f = dup(f);
    let f = dup(f);
    let f = dup(f);
    let f = dup(f);
    let f = dup(f);

    println!("Type size was at least {}", f(1));
}
