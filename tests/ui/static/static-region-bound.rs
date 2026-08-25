fn id<T>(x: T) -> T { x }

fn f<T:'static>(_: T) {}

fn main() {

    let x: Box<_> = Box::new(3);
    f(x);

    let x = &id(3); //~ ERROR temporary value dropped while borrowed
    f(x);
}
