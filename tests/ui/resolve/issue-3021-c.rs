fn siphash<T>() {

    trait U {
        fn g(&self, x: T) -> T;  //~ ERROR can't use generic parameters from outer function
        //~^ ERROR can't use generic parameters from outer function
    }
}

fn main() {}
