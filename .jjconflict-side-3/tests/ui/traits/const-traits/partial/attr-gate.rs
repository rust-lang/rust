trait A {
    #[rustc_non_const_trait_method]
    //~^ ERROR: use of an internal attribute
    fn a();
}

fn main() {}
