use run_make_support::rustdoc;
use run_make_support::serde_json::{self, Value};

const REPEAT: usize = 150;

fn main() {
    let deeply_nested_code =
        format!("pub type Foo = {}u128{};", "Box<".repeat(REPEAT), ">".repeat(REPEAT));
    let output = rustdoc()
        .stdin_buf(deeply_nested_code)
        .args(["-", "-Zunstable-options", "-wjson", "-o-"])
        .run()
        .stdout();
    let parsed = serde_json::from_slice::<Value>(&output).expect("failed to parse JSON");

    assert!(max_depth(&parsed) < 15);
}

fn max_depth(v: &Value) -> usize {
    match v {
        Value::Null => 1,
        Value::Bool(_) => 1,
        Value::Number(_) => 1,
        Value::String(_) => 1,
        Value::Array(a) => a.iter().map(max_depth).max().unwrap_or(0) + 1,
        Value::Object(o) => o.values().map(max_depth).max().unwrap_or(0) + 1,
    }
}
