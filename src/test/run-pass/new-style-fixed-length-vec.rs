use core::io::println;

static FOO: [int, ..3] = [1, 2, 3];

fn main() {
    println(fmt!("%d %d %d", FOO[0], FOO[1], FOO[2]));
}



