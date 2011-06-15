

type foo[T] = tup(T);

type bar[T] = foo[T];

fn takebar[T](&bar[T] b) { }

fn main() { }