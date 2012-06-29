// Issue #1821 - Don't recurse trying to typecheck this
enum t {
    foo(~[t])
}
fn main() {}