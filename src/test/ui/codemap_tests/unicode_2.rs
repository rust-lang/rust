#![feature(non_ascii_idents)]

fn main() {
    let _ = ("a̐éö̲", 0u7); //~ ERROR invalid width
    let _ = ("아あ", 1i42); //~ ERROR invalid width
    let _ = a̐é; //~ ERROR cannot find
}
