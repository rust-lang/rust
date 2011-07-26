

fn main() {
    // This just tests whether the vec leaks its members.

    let vec[@rec(int x, int y)] pvec = [@rec(x=1, y=2),
                                        @rec(x=3, y=4),
                                        @rec(x=5, y=6)];
}