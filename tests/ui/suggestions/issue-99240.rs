fn fmt(it: &(std::cell::Cell<Option<impl FnOnce()>>,)) {
    (it.0.take())()
    //~^ ERROR expected function
}

fn main() {}
