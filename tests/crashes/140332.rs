//@ known-bug: #140332

static mut S: [i8] = ["Some thing"; 1];

fn main() {
    assert_eq!(S, [0; 1]);
}
