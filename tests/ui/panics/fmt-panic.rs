//@ run-fail
//@ check-run-results:meh
//@ ignore-emscripten no processes

fn main() {
    let str_var: String = "meh".to_string();
    panic!("{}", str_var);
}
