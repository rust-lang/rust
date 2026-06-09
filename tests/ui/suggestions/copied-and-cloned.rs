//@ run-rustfix

fn expect<T>(_: T) {}

struct Issue114925 {
    x: Option<String>,
}

fn issue_114925(lol: &mut Issue114925, x: Option<&String>) {
    lol.x = x.clone();
    //~^ ERROR mismatched types
    //~| HELP use `Option::cloned` to clone the value inside the `Option`
}

fn main() {
    let x = Some(&());
    expect::<Option<()>>(x);
    //~^ ERROR mismatched types
    //~| HELP use `Option::copied` to copy the value inside the `Option`
    let x = Ok(&());
    expect::<Result<(), ()>>(x);
    //~^ ERROR mismatched types
    //~| HELP use `Result::copied` to copy the value inside the `Result`
    let s = String::new();
    let x = Some(&s);
    expect::<Option<String>>(x);
    //~^ ERROR mismatched types
    //~| HELP use `Option::cloned` to clone the value inside the `Option`
    let x = Ok(&s);
    expect::<Result<String, ()>>(x);
    //~^ ERROR mismatched types
    //~| HELP use `Result::cloned` to clone the value inside the `Result`

    let s = String::new();
    let x = Some(s.clone());
    let y = Some(&s);
    println!("{}", x == y);
    //~^ ERROR mismatched types
    //~| HELP use `Option::cloned` to clone the value inside the `Option`
    //FIXME(#114050) ~| HELP use `Option::as_ref` to convert `Option<String>` to `Option<&String>`

    let mut s = ();
    let x = Some(s);
    let y = Some(&mut s);
    println!("{}", x == y);
    //~^ ERROR mismatched types
    //~| HELP use `Option::copied` to copy the value inside the `Option`

    let mut s = String::new();
    let x = Some(s.clone());
    let y = Some(&mut s);
    println!("{}", x == y);
    //~^ ERROR mismatched types
    //~| HELP use `Option::cloned` to clone the value inside the `Option`

    issue_114925(&mut Issue114925 { x: None }, None);
}
