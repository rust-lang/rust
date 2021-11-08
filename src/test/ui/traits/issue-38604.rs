trait Q<T:?Sized> {}
trait Foo where u32: Q<Self> {
    fn foo(&self);
}

impl Q<()> for u32 {}
impl Foo for () {
    fn foo(&self) {
        println!("foo!");
    }
}

fn main() {
    let _f: Box<dyn Foo> = //~ ERROR `Foo` cannot be made into an object
        Box::new(()); //~ ERROR `Foo` cannot be made into an object
}
