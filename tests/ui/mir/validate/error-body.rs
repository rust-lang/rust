//@ compile-flags: -Zvalidate-mir

fn _test() {
    let x = || 45;
    missing();
    //~^ ERROR cannot find function `missing` in this scope
}

fn main() {}
