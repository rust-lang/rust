type Value = String;

fn main() {
    let f = |&Value::String(_)| (); //~ ERROR no associated function or constant named

    let vec: Vec<Value> = Vec::new();
    vec.last().map(f);
}
