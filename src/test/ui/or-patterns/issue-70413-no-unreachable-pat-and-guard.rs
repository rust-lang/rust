// check-pass

#![deny(unreachable_patterns)]

#![feature(or_patterns)]
fn main() {
    match (3,42) {
        (a,_) | (_,a) if a > 10 => {println!("{}", a)}
        _ => ()
    }

    match Some((3,42)) {
        Some((a, _)) | Some((_, a)) if a > 10 => {println!("{}", a)}
        _ => ()

    }

    match Some((3,42)) {
        Some((a, _) | (_, a)) if a > 10 => {println!("{}", a)}
        _ => ()
    }
}
