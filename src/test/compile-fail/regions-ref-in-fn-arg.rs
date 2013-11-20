fn arg_item(~ref x: ~int) -> &'static int {
    x //~^ ERROR borrowed value does not live long enough
}

fn with<R>(f: |~int| -> R) -> R { f(~3) }

fn arg_closure() -> &'static int {
    with(|~ref x| x) //~ ERROR borrowed value does not live long enough
}

fn main() {}
