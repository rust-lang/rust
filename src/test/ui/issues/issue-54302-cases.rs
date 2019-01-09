trait Mirror {
    type Image;
    fn coerce(self) -> Self::Image;
}

impl<T> Mirror for T {
    type Image = T;
    fn coerce(self) -> Self { self }
}

trait Foo<'x, T> {
    fn foo(self) -> &'x T;
}

impl<'s, 'x, T: 'x> Foo<'x, T> for &'s T where &'s T: Foo2<'x, T> {
    fn foo(self) -> &'x T { self.foo2() }
}

trait Foo2<'x, T> {
    fn foo2(self) -> &'x T;
}

// example 1 - fails leak check
impl<'x> Foo2<'x, u32> for &'x u32
{
    fn foo2(self) -> &'x u32 { self }
}

// example 2 - OK with this issue
impl<'x, 'a: 'x> Foo2<'x, i32> for &'a i32
{
    fn foo2(self) -> &'x i32 { self }
}

// example 3 - fails due to issue #XYZ + Leak-check
impl<'x, T> Foo2<'x, u64> for T
    where T: Mirror<Image=&'x u64>
{
    fn foo2(self) -> &'x u64 { self.coerce() }
}

// example 4 - fails due to issue #XYZ
impl<'x, 'a: 'x, T> Foo2<'x, i64> for T
    where T: Mirror<Image=&'a i64>
{
    fn foo2(self) -> &'x i64 { self.coerce() }
}


trait RefFoo<T> {
    fn ref_foo(&self) -> &'static T;
}

impl<T> RefFoo<T> for T where for<'a> &'a T: Foo<'static, T> {
    fn ref_foo(&self) -> &'static T {
        self.foo()
    }
}


fn coerce_lifetime1(a: &u32) -> &'static u32
{
    <u32 as RefFoo<u32>>::ref_foo(a)
    //~^ ERROR not general enough
}

fn coerce_lifetime2(a: &i32) -> &'static i32
{
    <i32 as RefFoo<i32>>::ref_foo(a)
    //~^ ERROR not general enough
}

fn coerce_lifetime3(a: &u64) -> &'static u64
{
    <u64 as RefFoo<u64>>::ref_foo(a)
    //~^ ERROR not general enough
}

fn coerce_lifetime4(a: &i64) -> &'static i64
{
    <i64 as RefFoo<i64>>::ref_foo(a)
    //~^ ERROR not general enough
}

fn main() {}
