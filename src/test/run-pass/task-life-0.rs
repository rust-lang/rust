use std;
import std::task;
fn main() { task::_spawn(bind child("Hello")); }

fn child(s: str) {

}
