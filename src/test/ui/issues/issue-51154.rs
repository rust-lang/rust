fn foo<F: FnMut()>() {
    let _: Box<F> = Box::new(|| ());
    //~^ ERROR arguments to this function are incorrect
}

fn main() {}
