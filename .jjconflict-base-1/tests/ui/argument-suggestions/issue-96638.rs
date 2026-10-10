fn f(_: usize, _: &usize, _: usize) {}

fn arg<T>() -> T { todo!() }

fn main() {
    let x = arg(); // `x` must be inferred
    // The reference on `&x` is important to reproduce the ICE
    f(&x, ""); //~ ERROR function takes 3 arguments but 2 arguments were supplied
}
