#[link(name = "a", vers = "0.0")];
#[crate_type = "lib"];
#[legacy_exports];

trait i<T> { }

fn f<T>() -> i<T> {
    impl<T> (): i<T> { }

    () as i::<T>
}
