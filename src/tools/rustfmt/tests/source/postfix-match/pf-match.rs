#![feature(postfix_match)]

fn main() {
    let val = Some(42);

    val.match {
        Some(_) => 2,
        _ => 1
    };

    Some(2).match {
        Some(_) => true,
        None => false
    }.match {
        false => "ferris is cute",
        true => "I turn cats in to petted cats",
    }.match {
        _ => (),
    }
}