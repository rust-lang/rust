// xfail-fast
#[legacy_modes];

extern mod std;

fn main() { test00(); }

fn test00_start(c: pipes::Chan<int>, start: int, number_of_messages: int) {
    let mut i: int = 0;
    while i < number_of_messages { c.send(start + i); i += 1; }
}

fn test00() {
    let mut r: int = 0;
    let mut sum: int = 0;
    let p = pipes::PortSet();
    let number_of_messages: int = 10;

    let c = p.chan();
    do task::spawn {
        test00_start(c, number_of_messages * 0, number_of_messages);
    }
    let c = p.chan();
    do task::spawn {
        test00_start(c, number_of_messages * 1, number_of_messages);
    }
    let c = p.chan();
    do task::spawn {
        test00_start(c, number_of_messages * 2, number_of_messages);
    }
    let c = p.chan();
    do task::spawn {
        test00_start(c, number_of_messages * 3, number_of_messages);
    }

    let mut i: int = 0;
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

    assert (sum == number_of_messages * 4 * (number_of_messages * 4 - 1) / 2);
}
