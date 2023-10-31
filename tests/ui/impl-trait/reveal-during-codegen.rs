// build-pass
// revisions: current next
//[next] compile-flags: -Ztrait-solver=next

fn test() -> Option<impl Sized> {
    Some("")
}

fn main() {
    test();
}
