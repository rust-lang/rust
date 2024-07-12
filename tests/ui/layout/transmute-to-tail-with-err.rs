trait Trait<T> {}

struct Bar(Box<dyn Trait<T>>);
//~^ ERROR cannot find type `T`

fn main() {
    let x: Bar = unsafe { std::mem::transmute(()) };
}
