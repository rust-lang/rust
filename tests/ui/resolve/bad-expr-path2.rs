mod m1 {
    pub mod arguments {}
}

fn main(arguments: Vec<String>) { //~ ERROR `main` function has wrong type
    log(debug, m1::arguments);
    //~^ ERROR cannot find function `log`
    //~| ERROR cannot find value `debug`
    //~| ERROR expected value, found module `m1::arguments`
}
