enum color { rgb(isize, isize, isize), rgba(isize, isize, isize, isize), }

fn main() {
    let red: color = color::rgb(255, 0, 0);
    match red {
      color::rgb(r, g, b) => { println!("rgb"); }
      color::hsl(h, s, l) => { println!("hsl"); }
      //~^ ERROR no variant
    }
}
