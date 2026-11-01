trait Trait {}

struct Chars;
impl Trait for Chars {}

struct FlatMap<T>(T);
impl<T: Trait> std::fmt::Debug for FlatMap<T> {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unimplemented!()
    }
}

fn lol() {
    format_args!("{:?}", FlatMap(&Chars));
    //~^ ERROR the trait bound `&Chars: Trait` is not satisfied [E0277]
}

fn main() {}
