//@ check-pass

// RSplit<T, P> previously required T: Clone in order to be Clone

struct NotClone;

fn main() {
    let elements = [NotClone, NotClone, NotClone];
    let rsplit = elements.rsplit(|_| false);
    rsplit.clone();
}
