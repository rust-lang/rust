// regression test for #26376

trait Foo {
    fn foo(&self) -> [u8];
}

fn foo(f: Option<&dyn Foo>) {
    if let Some(f) = f {
        let _ = f.foo();
        //~^ ERROR cannot move a value of type [u8]: the size of [u8] cannot be statically determined
    }
}

fn main() { foo(None) }
