// issue #500

fn main() {
    let x = true;
    let y = 1;
    let z = x + y;
    //~^ ERROR cannot add `{integer}` to `bool`
}
