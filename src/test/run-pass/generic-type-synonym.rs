

type foo<T> = {a: T};

type bar<T> = foo<T>;

fn takebar<T>(b: &bar<T>) { }

fn main() { }
