// issue: 114131

fn main() {
    let hello = len(vec![]);
    //~^ ERROR cannot find function `len` in this scope
}
