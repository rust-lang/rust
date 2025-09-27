fn main() {
    //~^ HELP consider importing this function
    //~| HELP consider importing this function
    //~| HELP consider importing this function
    //~| HELP consider importing this function
    let x = 2;
    let y = 4;

    max(x, y);
    //~^ ERROR cannot find function `max` in this scope
    //~| HELP you may have meant to use the method syntax
    let _ = min(x, y);
    //~^ ERROR cannot find function `min` in this scope
    //~| HELP you may have meant to use the method syntax
    println!("{}", min(43, 43));
    //~^ ERROR cannot find function `min` in this scope
    //~| HELP you may have meant to use the method syntax
    let _ = vec![max(f(), g())];
    //~^ ERROR cannot find function `max` in this scope
    //~| HELP you may have meant to use the method syntax
}

const fn f() -> u32 {
    4
}

const fn g() -> u32 {
    2
}
