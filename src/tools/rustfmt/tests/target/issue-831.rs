fn main() {
    let y = a.iter().any(|x| {
        println!("a");
    }) || b.iter().any(|x| {
        println!("b");
    }) || c.iter().any(|x| {
        println!("c");
    });
}
