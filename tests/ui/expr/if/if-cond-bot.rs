// run-fail
//@error-in-other-file:quux
//@ignore-target-emscripten no processes

fn my_err(s: String) -> ! {
    println!("{}", s);
    panic!("quux");
}

fn main() {
    if my_err("bye".to_string()) {
    }
}
