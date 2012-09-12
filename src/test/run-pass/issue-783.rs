extern mod std;
use comm::*;
use task::*;

fn a() {
    fn doit() {
        fn b(c: Chan<Chan<int>>) {
            let p = Port();
            send(c, Chan(p));
        }
        let p = Port();
        let ch = Chan(p);
        spawn(|| b(ch) );
        recv(p);
    }
    let mut i = 0;
    while i < 100 {
        doit();
        i += 1;
    }
}

fn main() {
    for iter::repeat(100u) {
        spawn(|| a() );
    }
}
