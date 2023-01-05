// run-pass

fn foo<T:Clone>(x: &T) -> T{
    match x {
        &ref a => (*a).clone()
    }
}

pub fn main() {
    assert_eq!(foo(&3), 3);
    assert_eq!(foo(&'a'), 'a');
}
