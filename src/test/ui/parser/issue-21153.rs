trait MyTrait<T>: Iterator {
    //~^ ERROR missing `fn`, `type`, `const`, or `static` for item declaration
    Item = T;
}

fn main() {}
