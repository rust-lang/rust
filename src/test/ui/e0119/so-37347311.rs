// Ref: https://stackoverflow.com/q/37347311

trait Storage {
    type Error;
}

enum MyError<S: Storage> {
    StorageProblem(S::Error),
}

impl<S: Storage> From<S::Error> for MyError<S> { //~ ERROR conflicting implementations
    fn from(error: S::Error) -> MyError<S> {
        MyError::StorageProblem(error)
    }
}

fn main() {}
