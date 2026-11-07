trait Trait {
    fn outer(&self) {
        fn inner(_: &Self) {
            //~^ ERROR can't use `Self` from outer item
        }
    }
}

fn main() { }
