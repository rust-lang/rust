thread_local! {
    //~^ ERROR: `#[used(linker)]` is currently unstable [E0658]

    #[used(linker)] //~ ERROR: the `used` attribute cannot be used on constants
    pub static BAZ: () = ();
}

fn main() {}
