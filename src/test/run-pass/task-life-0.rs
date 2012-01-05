use std;
import task;
fn main() {
    task::spawn {|| child("Hello"); };
}

fn child(&&s: str) {

}
