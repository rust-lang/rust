trait Foo { fn foo(&self) {} }
impl<T> Foo for T {}

fn main()
{
    let a : *const str = "hello";
    let _ = a as *const dyn Foo; //~ ERROR the size for values of type
}
