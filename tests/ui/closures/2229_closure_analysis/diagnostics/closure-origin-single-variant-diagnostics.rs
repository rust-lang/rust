//@ edition:2021


enum SingleVariant {
    Point(i32, i32),
}

fn main() {
    let mut point = SingleVariant::Point(10, -10);

    let c = || {
        let SingleVariant::Point(ref mut x, _) = point;
        *x += 1;
    };

    let b = c;
    let a = c; //~ ERROR use of moved value: `c` [E0382]
}
