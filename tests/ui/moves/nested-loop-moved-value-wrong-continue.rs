fn foo() {
    let foos = vec![String::new()];
    let bars = vec![""];
    let mut baz = vec![];
    let mut qux = vec![];
    for foo in foos { for bar in &bars { if foo == *bar {
    //~^ NOTE this reinitialization might get skipped
    //~| NOTE move occurs because `foo` has type `String`
    //~| NOTE inside of this loop
    //~| HELP consider moving the expression out of the loop
    //~| NOTE in this expansion of desugaring of `for` loop
    //~| NOTE
    //~| NOTE
        baz.push(foo);
        //~^ NOTE value moved here
        //~| HELP consider cloning the value
        continue;
        //~^ NOTE verify that your loop breaking logic is correct
        //~| NOTE this `continue` advances the loop at $DIR/nested-loop-moved-value-wrong-continue.rs:6:23
    } }
    qux.push(foo);
    //~^ ERROR use of moved value
    //~| NOTE value used here
    }
}

fn main() {
    let foos = vec![String::new()];
    let bars = vec![""];
    let mut baz = vec![];
    let mut qux = vec![];
    for foo in foos {
    //~^ NOTE this reinitialization might get skipped
    //~| NOTE move occurs because `foo` has type `String`
    //~| NOTE
        for bar in &bars {
        //~^ NOTE inside of this loop
        //~| HELP consider moving the expression out of the loop
        //~| NOTE in this expansion of desugaring of `for` loop
        //~| NOTE
            if foo == *bar {
                baz.push(foo);
                //~^ NOTE value moved here
                //~| HELP consider cloning the value
                continue;
                //~^ NOTE verify that your loop breaking logic is correct
                //~| NOTE this `continue` advances the loop at line 36
            }
        }
        qux.push(foo);
        //~^ ERROR use of moved value
        //~| NOTE value used here
    }
}
