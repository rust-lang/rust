use std;
fn main() {
    task::spawn(|| child(~"Hello") );
}

fn child(&&s: ~str) {

}
