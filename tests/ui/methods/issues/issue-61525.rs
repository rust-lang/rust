pub trait Example {
    fn query<Q>(self, q: Q);
}

impl Example for i32 {
    fn query<Q>(self, _: Q) {
        unimplemented!()
    }
}

mod nested {
    use super::Example;
    fn example() {
        1.query::<dyn ToString>("")
        //~^ ERROR the size for values of type `dyn ToString` cannot be known at compilation time
        //~| ERROR mismatched types
    }
}

fn main() {}
