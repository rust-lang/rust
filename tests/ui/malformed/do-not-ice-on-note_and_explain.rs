struct A<B>(B);

impl<B> A<B> {
    fn d() {
        fn d() {
            Self(1)
            //~^ ERROR can't reference `Self` constructor from outer item
        }
    }
}

fn main() {}
