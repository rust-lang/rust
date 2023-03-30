// revisions: old new
//[new] compile-flags: -Ztrait-solver=next
//[old] check-pass
//[new] known-bug: #109764


pub struct Bar
where
    for<'a> &'a mut Self:;

fn main() {}
