// Test that various patterns also enforce types.

#![feature(nll)]

fn main() {
    let _: Vec<&'static String> = vec![&String::new()];
    //~^ ERROR borrowed value does not live long enough [E0597]

    let (_, a): (Vec<&'static String>, _) = (vec![&String::new()], 44);
    //~^ ERROR borrowed value does not live long enough [E0597]

    let (_a, b): (Vec<&'static String>, _) = (vec![&String::new()], 44);
    //~^ ERROR borrowed value does not live long enough [E0597]
}
