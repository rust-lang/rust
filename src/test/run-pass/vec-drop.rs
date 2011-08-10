

fn main() {
    // This just tests whether the vec leaks its members.

    let pvec: [@{x: int, y: int}] =
        ~[@{x: 1, y: 2}, @{x: 3, y: 4}, @{x: 5, y: 6}];
}
