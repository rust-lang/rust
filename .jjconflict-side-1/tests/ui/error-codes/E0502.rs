fn bar(x: &mut i32) {}
fn foo(a: &mut i32) {
    let ref y = a;
    bar(a); //~ ERROR E0502
    y.use_ref();
}

fn main() {
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
