thread_local! {
    #[cfg_attr(true, rustc_dummy = 17)] //~ ERROR: use of an internal attribute [E0658]
    pub static BAR: () = ();
}

fn main() {}
