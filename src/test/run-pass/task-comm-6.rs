use std;
import pipes;
import pipes::send;
import pipes::chan;
import pipes::recv;

fn main() { test00(); }

fn test00() {
    let mut r: int = 0;
    let mut sum: int = 0;
    let p = pipes::port_set();
    let c0 = p.chan();
    let c1 = p.chan();
    let c2 = p.chan();
    let c3 = p.chan();
    let number_of_messages: int = 1000;
    let mut i: int = 0;
    while i < number_of_messages {
        c0.send(i + 0);
        c1.send(i + 0);
        c2.send(i + 0);
        c3.send(i + 0);
        i += 1;
    }
    i = 0;
    while i < number_of_messages {
        r = p.recv();
        sum += r;
        r = p.recv();
        sum += r;
        r = p.recv();
        sum += r;
        r = p.recv();
        sum += r;
        i += 1;
    }
    assert (sum == 1998000);
    // assert (sum == 4 * ((number_of_messages *
    //                   (number_of_messages - 1)) / 2));

}
