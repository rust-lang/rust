fn assert_all<F, T>(_f: F)
where
    F: FnMut(&String) -> T,
{
}

fn id(x: &String) -> &String {
    x
}

fn main() {
    assert_all::<_, &String>(id);
    //~^ ERROR implementation of `FnMut` is not general enough
}
