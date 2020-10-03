macro_rules! test {
    ($a, $b) => {
        //~^ ERROR missing fragment
        //~| ERROR missing fragment
        //~| WARN this was previously accepted
        ()
    };
}

fn main() {
    test!()
}
