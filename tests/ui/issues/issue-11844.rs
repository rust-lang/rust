fn main() {
    let a = Some(Box::new(1));
    match a {
        Ok(a) => //~ ERROR: mismatched types
            println!("{}",a),
        None => panic!()
    }
}
