use test_float_parse::validate;

fn main() {
    for i in 0..(1 << 19) {
        validate(&i.to_string());
    }
}
