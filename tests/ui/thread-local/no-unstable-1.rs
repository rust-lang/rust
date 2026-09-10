thread_local! {
    #[rustc_dummy = 17] //~ ERROR: use of an internal attribute [E0658]
    pub static FOO: () = ();
}

fn main() {}
