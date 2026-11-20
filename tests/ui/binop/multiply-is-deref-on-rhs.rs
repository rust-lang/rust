pub fn test(y: &i32) {
    let x;
    x = ()
    *y //~ ERROR: can't multiply `()` by `&i32`
}

fn main() {}
