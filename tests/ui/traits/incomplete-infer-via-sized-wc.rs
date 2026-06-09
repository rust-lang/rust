//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Exercises change in <https://github.com/rust-lang/rust/pull/138176>.

struct MaybeSized<T: ?Sized>(T);

fn is_sized<T: Sized>() -> Box<T> { todo!() }

fn foo<T: ?Sized>()
where
    MaybeSized<T>: Sized,
{
    is_sized::<MaybeSized<_>>();
    //~^ ERROR type annotations needed
}

fn main() {}
