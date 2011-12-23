// error-pattern:attempted access of field some_field_name on type [int]
// issue #367

fn f() {
    let v = [1];
    log(debug, v.some_field_name); //type error
}

fn main() { }
