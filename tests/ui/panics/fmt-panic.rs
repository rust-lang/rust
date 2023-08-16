// run-fail
//@error-in-other-file:meh
//@ignore-target-emscripten no processes

fn main() {
    let str_var: String = "meh".to_string();
    panic!("{}", str_var);
}
