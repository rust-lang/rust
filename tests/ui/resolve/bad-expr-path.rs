mod m1 {}

fn main(arguments: Vec<String>) { //~ ERROR `main` function has wrong type
    log(debug, m1::arguments);
    //~^ ERROR cannot find function `log`
    //~| ERROR cannot find value `debug`
    //~| ERROR cannot find value `arguments`
}
