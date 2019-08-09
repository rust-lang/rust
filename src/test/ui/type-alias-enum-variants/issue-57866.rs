// build-pass (FIXME(62277): could be check-pass?)

enum Outer<T> {
    A(T)
}

enum Inner {
    A(i32)
}

type OuterAlias = Outer<Inner>;

fn ice(x: OuterAlias) {
    // Fine
    match x {
        OuterAlias::A(Inner::A(_)) => (),
    }
    // Not fine
    match x {
        OuterAlias::A(Inner::A(y)) => (),
    }
}

fn main() {}
