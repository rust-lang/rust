trait ServerFn {
    type Output;
    fn run_body() -> impl Sized;
}
struct MyServerFn {}

macro_rules! f {
    () => {
        impl ServerFn for MyServerFn {
            type Output = ();
            fn run_body() -> impl Sized {}
        }
    };
}

f! {}

fn problem<T: ServerFn<Output = i64>>(_: T) {}

fn main() {
    problem(MyServerFn {});
    //~^ ERROR type mismatch resolving `<MyServerFn as ServerFn>::Output == i64`
}
