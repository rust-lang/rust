use std;
import std::comm;
import std::comm::send;
import comm::chan;
import comm::recv;

fn main() { test00(); }

fn test00() {
    let r: int = 0;
    let sum: int = 0;
    let p = comm::port();
    let c0 = chan(p);
    let c1 = chan(p);
    let c2 = chan(p);
    let c3 = chan(p);
    let number_of_messages: int = 1000;
    let i: int = 0;
    while i < number_of_messages {
        send(c0, i + 0);
        send(c1, i + 0);
        send(c2, i + 0);
        send(c3, i + 0);
        i += 1;
    }
    i = 0;
    while i < number_of_messages {
        r = recv(p);
        sum += r;
        r = recv(p);
        sum += r;
        r = recv(p);
        sum += r;
        r = recv(p);
        sum += r;
        i += 1;
    }
    assert (sum == 1998000);
    // assert (sum == 4 * ((number_of_messages *
    //                   (number_of_messages - 1)) / 2));

}
