use std::collections::BTreeSet;
use std::collections::VecDeque;

fn main() {
    let mut x = BTreeSet::new();
    x.push(1); //~ ERROR E0599
    //~^ HELP you might have meant to use `insert`
    let mut x = Vec::new();
    x.push_back(1); //~ ERROR E0599
    //~^ HELP you might have meant to use `push`
    let mut x = VecDeque::new();
    x.push(1); //~ ERROR E0599
    //~^ HELP you might have meant to use `push_back`
    let mut x = vec![1, 2, 3];
    x.length(); //~ ERROR E0599
    //~^ HELP you might have meant to use `len`
    x.size(); //~ ERROR E0599
    //~^ HELP you might have meant to use `len`
    //~| HELP there is a method with a similar name
    String::new().push(""); //~ ERROR E0308
}
