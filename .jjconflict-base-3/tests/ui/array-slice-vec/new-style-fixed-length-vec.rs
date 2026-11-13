//@ run-pass

static FOO: [isize; 3] = [1, 2, 3];

pub fn main() {
    println!("{} {} {}", FOO[0], FOO[1], FOO[2]);
}
