// run-pass
// Test that we do not error out because of a (False) ambiguity
// between the builtin rules for Sized and the where clause. Issue
// #20959.

// pretty-expanded FIXME #23616

fn foo<K>(x: Option<K>)
    where Option<K> : Sized
{
    let _y = x;
}

fn main() {
    foo(Some(22));
}
