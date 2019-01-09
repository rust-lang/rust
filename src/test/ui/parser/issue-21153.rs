trait MyTrait<T>: Iterator { //~ ERROR missing `fn`, `type`, or `const`
    Item = T;
}

fn main() {}
