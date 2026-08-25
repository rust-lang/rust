thread_local! {
    //~^ ERROR: `#[used(linker)]` is currently unstable [E0658]

    #[rustc_dummy = 17] //~ ERROR: use of an internal attribute [E0658]
    pub static FOO: () = ();

    #[cfg_attr(true, rustc_dummy = 17)] //~ ERROR: use of an internal attribute [E0658]
    pub static BAR: () = ();

    #[used(linker)] //~ ERROR: the `used` attribute cannot be used on constants
    pub static BAZ: () = ();
}

fn main() {}
