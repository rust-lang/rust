//@ compile-flags: -O
//@ run-pass

pub enum Value {
    V0(i32),
    V1,
}

fn set_discriminant(val: &mut Value) -> Value {
    let val_alias: &Value = get(val);
    let mut stop = false;
    let Value::V0(v) = *val_alias else {
        unreachable!();
    };
    loop {
        let v = Value::V0(v);
        if stop {
            return v;
        }
        stop = true;
        *val = Value::V1;
    }
}

fn main() {
    let mut v = Value::V0(1);
    let v = set_discriminant(&mut v);
    assert!(matches!(v, Value::V0(1)));
}

#[inline(never)]
fn get<T>(v: &T) -> &T {
    v
}
