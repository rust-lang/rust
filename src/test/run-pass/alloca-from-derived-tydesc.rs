// xfail-stage0

tag option[T] {
    some(T);
    none;
}

type r[T] = rec(mutable (option[T])[] v);

fn f[T]() -> T[] {
    ret ~[];
}

fn main() {
    let r[int] r = rec(mutable v=~[]);
    r.v = f();
}
