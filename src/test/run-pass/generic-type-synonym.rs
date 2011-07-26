

type foo[T] = rec(T a);

type bar[T] = foo[T];

fn takebar[T](&bar[T] b) { }

fn main() { }