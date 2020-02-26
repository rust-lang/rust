trait MyTrait<T>: Iterator {
    Item = T;
    //~^ ERROR expected one of `!` or `::`, found `=`
}

fn main() {}
