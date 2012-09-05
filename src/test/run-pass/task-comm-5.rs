use std;

fn main() { test00(); }

fn test00() {
    let r: int = 0;
    let mut sum: int = 0;
    let (c, p) = pipes::stream();
    let number_of_messages: int = 1000;
    let mut i: int = 0;
    while i < number_of_messages { c.send(i + 0); i += 1; }
    i = 0;
    while i < number_of_messages { sum += p.recv(); i += 1; }
    assert (sum == number_of_messages * (number_of_messages - 1) / 2);
}
