//@ run-fail
//@ error-pattern:meh
//@ needs-subprocess

fn main() {
    let str_var: String = "meh".to_string();
    panic!("{}", str_var);
}
