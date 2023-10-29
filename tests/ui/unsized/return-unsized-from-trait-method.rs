// regression test for #26376

trait Foo {
    fn foo(&self) -> [u8];
}

fn foo(f: Option<&dyn Foo>) {
    if let Some(f) = f {
        let _ = f.foo();
        //~^ ERROR cannot be known at compilation time
    }
}

fn main() { foo(None) }
