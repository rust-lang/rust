trait Trait {
    fn outer(&self) {
        fn inner(_: &Self) {
            //~^ ERROR can't use type parameters from outer function
        }
    }
}

fn main() { }
