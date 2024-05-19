pub fn foo() {
    bar::<usize>();
}

pub fn bar<T>() {
    baz();
}

fn baz() {}
