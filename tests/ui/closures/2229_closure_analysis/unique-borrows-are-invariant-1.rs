// edition:2021

// regression test for #112056

fn extend_lifetime<'a, 'b>(x: &mut (&'a str,), y: &'b str) {
    let mut closure = |input| x.0 = input;
    //~^ ERROR: lifetime may not live long enough
    closure(y);
}

fn main() {
    let mut tuple = ("static",);
    {
        let x = String::from("temporary");
        extend_lifetime(&mut tuple, &x);
    }
    println!("{}", tuple.0);
}
