pub struct Outside<T>(T);

pub fn no_match<U, V>(a: Outside<U>, b: Outside<V>) -> (Outside<U>, Outside<V>) {
    unimplemented!();
}

pub fn no_match_2<U, V>(a: Outside<V>, b: Outside<U>) -> (Outside<U>, Outside<V>) {
    unimplemented!();
}

pub fn should_match_3<U>(a: Outside<U>, b: Outside<U>) -> (Outside<U>, Outside<U>) {
    unimplemented!();
}
