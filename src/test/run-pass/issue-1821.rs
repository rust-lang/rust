// Issue #1821 - Don't recurse trying to typecheck this


// pretty-expanded FIXME #23616

enum t {
    foo(Vec<t>)
}
pub fn main() {}
