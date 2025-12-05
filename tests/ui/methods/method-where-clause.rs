//@ run-pass
// Test that we can use method notation to call methods based on a
// where clause type, and not only type parameters.


trait Foo {
    fn foo(&self) -> i32;
}

impl Foo for Option<i32>
{
    fn foo(&self) -> i32 {
        self.unwrap_or(22)
    }
}

impl Foo for Option<u32>
{
    fn foo(&self) -> i32 {
        self.unwrap_or(22) as i32
    }
}

fn check<T>(x: Option<T>) -> (i32, i32)
    where Option<T> : Foo
{
    let y: Option<T> = None;
    (x.foo(), y.foo())
}

fn main() {
    assert_eq!(check(Some(23u32)), (23, 22));
    assert_eq!(check(Some(23)), (23, 22));
}
