#[link(name = "a", vers = "0.0")];
#[crate_type = "lib"];

trait i<T> { }

fn f<T>() -> i<T> {
    impl <T> of i<T> for () { }

    () as i::<T>
}
