struct X<F> where F: FnOnce() + 'static + Send {
    field: F,
}

fn foo<F>(blk: F) -> X<F> where F: FnOnce() + 'static {
    //~^ ERROR `F` cannot be sent between threads safely
    return X { field: blk };
}

fn main() {
}
