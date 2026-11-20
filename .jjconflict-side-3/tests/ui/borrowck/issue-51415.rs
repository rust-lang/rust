//@ run-rustfix
// Regression test for #51415: match default bindings were failing to
// see the "move out" implied by `&s` below.

fn main() {
    let a = vec![String::from("a")];
    let opt = a.iter().enumerate().find(|(_, &s)| {
        //~^ ERROR cannot move out
        *s == String::from("d")
    }).map(|(i, _)| i);
    println!("{:?}", opt);
}
