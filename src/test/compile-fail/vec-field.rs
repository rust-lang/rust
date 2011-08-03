// error-pattern:attempted field access on type vec[int]
// issue #367

fn f() {
    let v = [1];
    log v.some_field_name; //type error
}

fn main() { }