trait MyTrait<T>: Iterator {
    Item = T;
    //~^ ERROR non-item in item list
}

fn main() {}
