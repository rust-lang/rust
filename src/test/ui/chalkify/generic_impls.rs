// compile-flags: -Z chalk

trait Foo { }

impl<T> Foo for (T, u32) { }

fn gimme<F: Foo>() { }

fn foo<T>() {
    gimme::<(T, u32)>();
    gimme::<(Option<T>, u32)>();
    gimme::<(Option<T>, f32)>(); //~ ERROR
}

fn main() {
    gimme::<(i32, u32)>();
    gimme::<(i32, f32)>(); //~ ERROR
}
