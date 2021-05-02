#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete

#[derive(Debug)]
struct Point {
    x: String,
    y: String,
}
fn main() {
    let mut c = {
        let mut p = Point {x: "1".to_string(), y: "2".to_string() };
        || {
           let x = &mut p.x;
           println!("{:?}", p);
            //~^ ERROR `p` does not live long enough
        }
    };
    c();
}
