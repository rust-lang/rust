
fn main() {
    // This just tests whether the vec leaks its members.
    let pvec: [mut @{a: int, b: int}]/~ =
        [mut @{a: 1, b: 2}, @{a: 3, b: 4}, @{a: 5, b: 6}]/~;
}
