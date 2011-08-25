use std;
import std::comm;

fn main() { test00(); }

fn test00() {
    let r: int = 0;
    let sum: int = 0;
    let p = comm::port();
    let c = comm::chan(p);
    let number_of_messages: int = 1000;
    let i: int = 0;
    while i < number_of_messages { comm::send(c, i + 0); i += 1; }
    i = 0;
    while i < number_of_messages { sum += comm::recv(p); i += 1; }
    assert (sum == number_of_messages * (number_of_messages - 1) / 2);
}
