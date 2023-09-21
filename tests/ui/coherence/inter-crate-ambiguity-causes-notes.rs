// revisions: old next
//[next] compile-flags: -Ztrait-solver=next

struct S;

impl From<()> for S {
    fn from(x: ()) -> Self {
        S
    }
}

impl<I> From<I> for S
//~^ ERROR conflicting implementations of trait
where
    I: Iterator<Item = ()>,
{
    fn from(x: I) -> Self {
        S
    }
}

fn main() {}
