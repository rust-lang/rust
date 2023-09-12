trait Trait {
    fn outer(&self) {
        fn inner(_: &Self) {
            //~^ ERROR can't use generic parameters from outer item
        }
    }
}

fn main() { }
