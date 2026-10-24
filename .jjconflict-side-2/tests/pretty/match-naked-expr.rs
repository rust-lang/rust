//@ pp-exact

fn main() {
    let x = Some(3);
    let _y =
        match x {
            Some(_) => "some(_)".to_string(),
            None => "none".to_string(),
        };
}
