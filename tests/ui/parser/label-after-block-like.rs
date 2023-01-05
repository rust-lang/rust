fn a() {
    if let () = () 'a {}
    //~^ ERROR labeled expression must be followed by `:`
    //~| ERROR expected `{`, found `'a`
}

fn b() {
    if true 'a {}
    //~^ ERROR labeled expression must be followed by `:`
    //~| ERROR expected `{`, found `'a`
}

fn c() {
    loop 'a {}
    //~^ ERROR labeled expression must be followed by `:`
    //~| ERROR expected `{`, found `'a`
}

fn d() {
    while true 'a {}
    //~^ ERROR labeled expression must be followed by `:`
    //~| ERROR expected `{`, found `'a`
}

fn e() {
    while let () = () 'a {}
    //~^ ERROR labeled expression must be followed by `:`
    //~| ERROR expected `{`, found `'a`
}

fn f() {
    for _ in 0..0 'a {}
    //~^ ERROR labeled expression must be followed by `:`
    //~| ERROR expected `{`, found `'a`
}

fn g() {
    unsafe 'a {}
    //~^ ERROR labeled expression must be followed by `:`
    //~| ERROR expected `{`, found `'a`
}

fn main() {}
