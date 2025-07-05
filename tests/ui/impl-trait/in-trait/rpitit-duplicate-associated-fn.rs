// issue#140796

trait Bar {
    fn method() -> impl Sized;
    fn method() -> impl Sized;
    //~^ ERROR: the name `method` is defined multiple times
}

fn main() {}
