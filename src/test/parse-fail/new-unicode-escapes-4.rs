// compile-flags: -Z parse-only -Z continue-parse-after-error

pub fn main() {
    let s = "\u{lol}";
     //~^ ERROR invalid character in unicode escape: l
}
