struct Foo<T> { t: &'static T }

fn main() {
    let x = 3;
    let _ = Foo { t: &x }; //~ ERROR `x` does not live long enough
}
