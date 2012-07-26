iface deref {
    fn get() -> int;
}

impl of deref for &int {
    fn get() -> int {
        *self
    }
}

fn with<R: deref>(f: fn(x: &int) -> R) -> int {
    f(&3).get()
}

fn return_it() -> int {
    with(|o| o)
    //~^ ERROR reference is not valid outside of its lifetime, &
    //~^^ ERROR reference is not valid outside of its lifetime, &
}

fn main() {
    let x = return_it();
    #debug["foo=%d", x];
}
