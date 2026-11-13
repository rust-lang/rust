//@ check-pass

enum Void {}

fn main() {
    let x: Void;
    match x {
        _ if { loop {} } => (),
    }
}
