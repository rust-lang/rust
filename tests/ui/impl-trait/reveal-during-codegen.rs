//@ build-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

fn test() -> Option<impl Sized> {
    Some("")
}

fn main() {
    test();
}
