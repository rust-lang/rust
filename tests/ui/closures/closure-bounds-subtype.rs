fn take_any<F>(_: F) where F: FnOnce() {
}

fn take_const_owned<F>(_: F) where F: FnOnce() + Sync + Send {
}

fn give_any<F>(f: F) where F: FnOnce() {
    take_any(f);
}

fn give_owned<F>(f: F) where F: FnOnce() + Send {
    take_any(f);
    take_const_owned(f); //~ ERROR `F` cannot be shared between threads safely [E0277]
}

fn main() {}
