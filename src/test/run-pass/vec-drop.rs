

fn main() {
    // This just tests whether the vec leaks its members.

    let vec[@tup(int, int)] pvec = [@tup(1, 2), @tup(3, 4), @tup(5, 6)];
}