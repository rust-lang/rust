pub enum Value {
    Float(Option<f64>),
}

fn main() {
    let _a = Value::Float( //~ ERROR this enum variant takes 1 argument but 4 arguments were supplied
        0,
        None,
        None,
        0,
    );
}
