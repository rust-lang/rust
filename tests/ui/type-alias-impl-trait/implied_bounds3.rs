//@ check-pass

fn foo<F>(_: F)
where
    F: 'static,
{
}

fn from<F: Send>(f: F) -> impl Send {
    f
}

fn bar<T>() {
    foo(from(|| ()))
}

fn main() {
}
