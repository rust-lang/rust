// -*- rust -*-

// error-pattern:fail
extern mod std;
use comm::Port;
use comm::recv;

fn grandchild() { fail ~"grandchild dies"; }

fn child() {
    let p = Port::<int>();
    task::spawn(|| grandchild() );
    let x = recv(p);
}

fn main() {
    let p = Port::<int>();
    task::spawn(|| child() );
    let x = recv(p);
}
