struct B<const I: u8>; //~ ERROR const generics are unstable

impl B<0> {
    fn bug() -> Self {
        panic!()
    }
}

fn main() {}
