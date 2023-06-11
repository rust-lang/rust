pub mod inner {
    pub fn i_am_here() {
        #[cfg(feature = "another one that doesn't exist")]
        loop {}
    }
}

fn main() {
    inner::i_am_here();
    // ensure that nothing bad happens when we are checking for cfgs
    inner::i_am_not(); //~ ERROR cannot find function
}
