//@ known-bug: #140823

struct Container<T> {
    data: T,
}

fn ice(callback: Box<dyn Fn(Container<&u8>)>) {
    let fails: Box<dyn Fn(&Container<&u8>)> = callback;
}
