#![feature(box_syntax)]

fn main() {
    let a = Some(box 1);
    match a {
        Ok(a) => //~ ERROR: mismatched types
            println!("{}",a),
        None => panic!()
    }
}
