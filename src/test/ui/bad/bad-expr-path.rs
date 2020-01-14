mod m1 {}

fn main(arguments: Vec<String>) {
    log(debug, m1::arguments);
    //~^ ERROR cannot find function `log` in this scope
    //~| ERROR cannot find value `debug` in this scope
    //~| ERROR cannot find value `arguments` in module `m1`
}
