fn main() {
    let foos = vec![String::new()];
    let bars = vec![""];
    let mut baz = vec![];
    let mut qux = vec![];
    for foo in foos {
    //~^ NOTE this reinitialization might get skipped
    //~| NOTE move occurs because `foo` has type `String`
        for bar in &bars {
        //~^ NOTE inside of this loop
        //~| HELP consider moving the expression out of the loop
        //~| NOTE in this expansion of desugaring of `for` loop
            if foo == *bar {
                baz.push(foo);
                //~^ NOTE value moved here
                //~| HELP consider cloning the value
                continue;
            }
        }
        qux.push(foo);
        //~^ ERROR use of moved value
        //~| NOTE value used here
    }
}
