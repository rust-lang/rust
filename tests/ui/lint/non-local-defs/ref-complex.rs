//@ check-pass

trait Trait<T> {}

fn main() {
    mod below {
        pub struct Type<T>(T);
    }
    struct InsideMain;
    trait HasFoo {}

    impl<T> Trait<InsideMain> for &Vec<below::Type<(InsideMain, T)>>
    where
        T: HasFoo
    {}
}
