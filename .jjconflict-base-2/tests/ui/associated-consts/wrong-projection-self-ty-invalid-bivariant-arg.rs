struct Fail<T>;
//~^ ERROR: type parameter `T` is never used

impl Fail<i32> {
    const C: () = ();
}

fn main() {
    Fail::<()>::C
    //~^ ERROR no associated function or constant named `C` found for struct `Fail<()>` in the current scope
}
