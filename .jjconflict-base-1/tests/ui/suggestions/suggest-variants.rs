#[derive(Debug)]
enum Shape {
  Square { size: i32 },
  Circle { radius: i32 },
}

struct S {
  x: usize,
}

fn main() {
    println!("My shape is {:?}", Shape::Squareee { size: 5});  //~ ERROR no variant named `Squareee`
    println!("My shape is {:?}", Shape::Circl { size: 5}); //~ ERROR no variant named `Circl`
    println!("My shape is {:?}", Shape::Rombus{ size: 5}); //~ ERROR no variant named `Rombus`
    Shape::Squareee; //~ ERROR no variant
    Shape::Circl; //~ ERROR no variant
    Shape::Rombus; //~ ERROR no variant
}

enum Color {
  Red,
  Green(()),
  Blue,
  Alpha{ a: u8 },
}

fn red() -> Result<Color, ()> {
  Ok(Color::Redd) //~ ERROR no variant
}

fn green() -> Result<Color, ()> {
  Ok(Color::Greenn(())) //~ ERROR no variant
}

fn blue() -> Result<Color, ()> {
  Ok(Color::Blu) //~ ERROR no variant
}

fn alpha() -> Result<Color, ()> {
  Ok(Color::Alph{ a: 255 }) //~ ERROR no variant
}
