fn update_node(foo: &mut Option<String>) {
    *foo = Some(foo.unwrap());
    //~^ ERROR cannot move out of `*foo`
}

fn main() {
    let mut my_option: Option<String> = Some("Hello".to_string());
    update_node(&mut my_option);
    println!("{:?}", my_option); // This will print: Some("Hello")
}
