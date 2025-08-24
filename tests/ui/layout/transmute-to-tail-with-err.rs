trait Trait<T> {}

struct Bar(Box<dyn Trait<T>>);
//~^ ERROR cannot find type `T` in this scope

fn main() {
    let x: Bar = unsafe { std::mem::transmute(()) };
    //~^ ERROR cannot transmute between types of different size
}
