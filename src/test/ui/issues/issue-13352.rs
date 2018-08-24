// ignore-cloudabi no std::process

fn foo(_: Box<FnMut()>) {}

fn main() {
    foo(loop {
        std::process::exit(0);
    });
    2_usize + (loop {});
    //~^ ERROR E0277
}
