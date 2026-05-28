trait Trait {
    type Gat<'a: 'b, 'b: 'c, 'c>: 'c;
}

fn get_func<'a, T: Trait>(_: &'a str) -> fn(T::Gat<'a, '_, 'static>) {
    loop {}
}

fn test<T: Trait>() {
    let func = get_func::<T>(&String::new()); //~ ERROR temporary value dropped
    drop(func);
}

fn main() {}
