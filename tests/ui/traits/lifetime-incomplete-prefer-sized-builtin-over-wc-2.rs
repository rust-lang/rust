//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Exercises change in <https://github.com/rust-lang/rust/pull/138176>.

trait Trait<T>: Sized {}
impl<T> Trait<T> for T {}

fn is_sized<T: Sized>() {}

fn normal_ref<'a, 'b, T>()
where
    &'a u32: Trait<T>,
{
    is_sized::<&'b u32>();
}

struct MyRef<'a, U: ?Sized = ()>(&'a u32, U);
fn my_ref<'a, 'b, T>()
where
    MyRef<'a>: Trait<T>,
{
    is_sized::<MyRef<'b>>();
}

fn main() {}
