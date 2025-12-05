//@ run-pass

#![feature(postfix_match)]

fn main() {
    1.match {
        2 => Some(0),
        _ => None,
    }.match {
        None => Ok(true),
        Some(_) => Err("nope")
    }.match {
        Ok(_) => (),
        Err(_) => panic!()
    }
}
