//@ check-pass
fn main() {
    let _: i32 = (match "" {
        "+" => ::std::ops::Add::add,
        "-" => ::std::ops::Sub::sub,
        "<" => |a,b| (a < b) as i32,
        _ => unimplemented!(),
    })(5, 5);
}
