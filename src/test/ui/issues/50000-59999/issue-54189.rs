fn bug() -> impl for <'r> Fn() -> &'r () { || { &() } }
//~^ ERROR binding for associated type `Output` references lifetime `'r`

fn main() {
    let f = bug();
}
