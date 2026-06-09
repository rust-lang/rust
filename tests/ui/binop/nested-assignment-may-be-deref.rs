pub fn bad(x: &mut bool) {
    if true
    *x = true {}
    //~^ ERROR cannot multiply `bool` by `&mut bool`
}

pub fn bad2(x: &mut bool) {
    let y: bool;
    y = true
    *x = true;
    //~^ ERROR cannot multiply `bool` by `&mut bool`
}

fn main() {}
