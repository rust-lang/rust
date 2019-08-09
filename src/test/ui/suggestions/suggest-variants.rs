#[derive(Debug)]
enum Shape {
  Square { size: i32 },
  Circle { radius: i32 },
}

struct S {
  x: usize,
}

fn main() {
    println!("My shape is {:?}", Shape::Squareee { size: 5});  //~ ERROR no variant `Squareee`
    println!("My shape is {:?}", Shape::Circl { size: 5}); //~ ERROR no variant `Circl`
    println!("My shape is {:?}", Shape::Rombus{ size: 5}); //~ ERROR no variant `Rombus`
    Shape::Squareee; //~ ERROR no variant
    Shape::Circl; //~ ERROR no variant
    Shape::Rombus; //~ ERROR no variant
}
