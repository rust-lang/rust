fn siphash<T>() {

    trait U {
        fn g(&self, x: T) -> T;  //~ ERROR can't use generic parameters from outer item
        //~^ ERROR can't use generic parameters from outer item
    }
}

fn main() {}
