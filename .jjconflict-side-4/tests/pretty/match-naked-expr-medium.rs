//@ pp-exact

fn main() {
    let x = Some(3);
    let _y =
        match x {
            Some(_) =>
                ["some(_)".to_string(), "not".to_string(), "SO".to_string(),
                        "long".to_string(), "string".to_string()],
            None =>
                ["none".to_string(), "a".to_string(), "a".to_string(),
                        "a".to_string(), "a".to_string()],
        };
}
