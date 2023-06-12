pub mod inner {
    #[cfg(FALSE)]
    mod gone {
        pub fn uwu() {}
    }

    #[cfg(FALSE)]
    pub use super::uwu;
    //~^ NOTE found an item that was configured out
}

fn main() {
    // There is no uwu at this path - no diagnostic.
    inner::uwu(); //~ ERROR cannot find function
    //~^ NOTE not found in `inner`
}
