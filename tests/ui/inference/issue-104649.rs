type Result<T, E = Error> = ::std::result::Result<T, E>;
struct Error;

trait ForEach {
    type Input;
    fn for_each<F, U>(self, f: F)
    where
        F: FnOnce(Self::Input) -> U;
}

impl<T> ForEach for A<T> {
    type Input = T;
    fn for_each<F, U>(self, f: F)
    where
        F: FnOnce(Self::Input) -> U,
    {
        todo!()
    }
}

struct A<T>(T);

fn main() {
    let a = A(Result::Ok(Result::Ok(()))); //~ ERROR type annotations needed
    a.for_each(|a: Result<_>| {
        let f = || match a {
            Ok(Ok(a)) => {}
            Ok(Err(a)) => {}
            Err(a) => {}
        };
    });
}
