// error-pattern:meh

fn main() {
    let str_var: String = "meh".to_string();
    panic!("{}", str_var);
}
