extern mod std;

fn siphash<T>() {

    trait t {
        fn g(x: T) -> T;  //~ ERROR attempt to use a type argument out of scope
        //~^ ERROR attempt to use a type argument out of scope
        //~^^ ERROR use of undeclared type name `T`
        //~^^^ ERROR use of undeclared type name `T`
    }
}

fn main() {}
