fn foo(x: isize) {
    fn mth() {
        fn bar() { log(debug, x); }
        //~^ ERROR can't capture dynamic environment in a fn item
        //~| ERROR cannot find value `debug` in this scope
        //~| ERROR cannot find function `log` in this scope
    }
}

fn main() { foo(2); }
