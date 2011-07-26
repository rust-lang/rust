
fn main() {
    // This just tests whether the vec leaks its members.
    let vec[mutable @rec(int a, int b)] pvec =
        [mutable @rec(a=1, b=2), @rec(a=3, b=4), @rec(a=5, b=6)];
}