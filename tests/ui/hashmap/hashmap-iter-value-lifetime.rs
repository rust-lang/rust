fn main() {
    let mut my_stuff = std::collections::HashMap::new();
    my_stuff.insert(0, 42);

    let (_, thing) = my_stuff.iter().next().unwrap();

    my_stuff.clear(); //~ ERROR cannot borrow

    println!("{}", *thing);
}
