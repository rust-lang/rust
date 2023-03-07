fn foo(_: Box<dyn FnMut()>) {}

fn main() {
    foo(loop {
        std::process::exit(0);
    });
    2_usize + (loop {});
    //~^ ERROR E0277
}
