//@ build-pass
//@ revisions: current next
//@[next] compile-flags: -Znext-solver

fn test() -> Option<impl Sized> {
    Some("")
}

fn main() {
    test();
}
