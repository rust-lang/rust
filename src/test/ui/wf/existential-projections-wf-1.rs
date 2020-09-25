// Taken from https://github.com/rust-lang/rust/issues/27675#issuecomment-696956878
trait Setup {
    type From: Copy;
}

fn copy<U: Setup + ?Sized>(from: &U::From) -> U::From {
    *from
}

pub fn copy_any<T>(t: &T) -> T {
    copy::<dyn Setup<From=T>>(t)
    //~^ ERROR the trait bound `T: Copy`
}

fn main() {
    let st = String::from("Hello");
    copy_any(&st);
}
