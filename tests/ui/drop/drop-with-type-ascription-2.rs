// run-pass

fn main() {
    let args = vec!["foobie", "asdf::asdf"];
    let arr: Vec<&str> = args[1].split("::").collect();
    assert_eq!(arr[0], "asdf");
    assert_eq!(arr[0], "asdf");
}
