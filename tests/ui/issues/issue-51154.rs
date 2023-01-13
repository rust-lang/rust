fn foo<F: FnMut()>() {
    let _: Box<F> = Box::new(|| ());
    //~^ ERROR mismatched types
}

fn main() {}
