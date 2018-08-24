// compile-flags: -Z parse-only

struct Rgb(u8, u8, u8);

fn main() {
    let _ = Rgb { 0, 1, 2 }; //~ ERROR expected identifier, found `0`
}
