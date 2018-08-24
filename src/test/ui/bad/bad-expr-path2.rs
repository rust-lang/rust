mod m1 {
    pub mod arguments {}
}

fn main(arguments: Vec<String>) { //~ ERROR main function has wrong type
    log(debug, m1::arguments);
    //~^ ERROR cannot find function `log` in this scope
    //~| ERROR cannot find value `debug` in this scope
    //~| ERROR expected value, found module `m1::arguments`
}
