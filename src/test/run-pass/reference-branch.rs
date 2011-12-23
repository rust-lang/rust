// Ensures that invalidating a reference in one branch doesn't
// influence other branches.

fn main() {
    let x = [];
    let &y = x;
    if true { x = [1]; }
    else { log(error, y); }
}
