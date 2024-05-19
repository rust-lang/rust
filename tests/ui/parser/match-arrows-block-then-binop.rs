fn main() {
    let _ = match 0 {
      0 => {
        0
      } + 5 //~ ERROR expected pattern, found `+`
    };
}
