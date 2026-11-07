fn foo(_f: impl Fn()) {}

fn bar() -> i32 {
    1
}

fn main() {
    foo(|| bar())
    //~^ ERROR mismatched types [E0308]
    //~| HELP consider using a semicolon here
}
