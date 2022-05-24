// check-pass

trait MyTrait<T> {}

fn foo<T>(f: impl MyTrait<T>) -> impl MyTrait<T> {
    f
}

fn main() {}
