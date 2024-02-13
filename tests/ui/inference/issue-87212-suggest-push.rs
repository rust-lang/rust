struct Thing;
fn t1() {
    let mut stuff: Vec<Thing> = Vec::new();
    // suggest push
    stuff.append(Thing); //~ ERROR mismatched types
}

fn t2() {
    let mut stuff = vec![];
    // suggest push
    stuff.append(Thing); //~ ERROR mismatched types
}

fn t3() {
    let mut stuff: Vec<i32> = Vec::new();
    // don't suggest push
    stuff.append(Thing); //~ ERROR mismatched types
}

fn main() {}
