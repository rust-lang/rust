// Issue #146515
use std::rc::Rc;

#[derive(Clone)]
struct ContainsRc<T, K> { //~ HELP `Clone` is not implemented
    value: Rc<(T, K)>,
}

fn clone_me<T, K>(x: &ContainsRc<T, K>) -> ContainsRc<T, K> {
    x.clone() //~ ERROR E0308
    //~^ HELP consider manually implementing `Clone`
}

#[derive(Clone)]
struct ContainsRcSingle<T> { //~ HELP `Clone` is not implemented
    value: Rc<T>,
}

fn clone_me_single<T>(x: &ContainsRcSingle<T>) -> ContainsRcSingle<T> {
    x.clone() //~ ERROR E0308
    //~^ HELP consider manually implementing `Clone`
}

fn main() {}
