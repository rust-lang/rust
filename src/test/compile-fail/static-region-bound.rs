#[feature(managed_boxes)];

fn f<T:'static>(_: T) {}

fn main() {
    let x = @3;
    f(x);
    let x = &3;
    f(x);   //~ ERROR instantiating a type parameter with an incompatible type
}
