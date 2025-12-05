//! Regression test for https://github.com/rust-lang/rust/issues/11844

fn main() {
    let a = Some(Box::new(1));
    match a {
        Ok(a) => //~ ERROR: mismatched types
            println!("{}",a),
        None => panic!()
    }
}
