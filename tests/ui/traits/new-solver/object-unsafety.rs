// compile-flags: -Ztrait-solver=next

trait Setup {
    type From: Copy;
}

fn copy<U: Setup + ?Sized>(from: &U::From) -> U::From {
    *from
}

pub fn copy_any<T>(t: &T) -> T {
    copy::<dyn Setup<From=T>>(t)
    //~^ ERROR the trait bound `dyn Setup<From = T>: Setup` is not satisfied
}

fn main() {
    let x = String::from("Hello, world");
    let y = copy_any(&x);
    println!("{y}");
}
