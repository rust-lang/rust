use std;
import task;
import comm;

fn main() { test00(); }

fn test00_start(c: comm::chan<int>, start: int, number_of_messages: int) {
    let i: int = 0;
    while i < number_of_messages { comm::send(c, start + i); i += 1; }
}

fn test00() {
    let r: int = 0;
    let sum: int = 0;
    let p = comm::port();
    let c = comm::chan(p);
    let number_of_messages: int = 10;

    let t0 = task::spawn_joinable {||
        test00_start(c, number_of_messages * 0, number_of_messages);
    };
    let t1 = task::spawn_joinable {||
        test00_start(c, number_of_messages * 1, number_of_messages);
    };
    let t2 = task::spawn_joinable {||
        test00_start(c, number_of_messages * 2, number_of_messages);
    };
    let t3 = task::spawn_joinable {||
        test00_start(c, number_of_messages * 3, number_of_messages);
    };

    let i: int = 0;
    while i < number_of_messages {
        r = comm::recv(p);
        sum += r;
        r = comm::recv(p);
        sum += r;
        r = comm::recv(p);
        sum += r;
        r = comm::recv(p);
        sum += r;
        i += 1;
    }

    task::join(t0);
    task::join(t1);
    task::join(t2);
    task::join(t3);

    assert (sum == number_of_messages * 4 * (number_of_messages * 4 - 1) / 2);
}
