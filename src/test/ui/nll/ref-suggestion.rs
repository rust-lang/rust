fn main() {
    let x = vec![1];
    let y = x;
    x; //~ ERROR use of moved value

    let x = vec![1];
    let mut y = x;
    x; //~ ERROR use of moved value

    let x = (Some(vec![1]), ());

    match x {
        (Some(y), ()) => {},
        _ => {},
    }
    x; //~ ERROR use of partially moved value
}
