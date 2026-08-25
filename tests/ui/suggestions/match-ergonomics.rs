fn main() {
    let x = vec![1i32];
    match &x[..] {
        [&v] => {}, //~ ERROR mismatched types
        _ => {},
    }
    match x {
        [&v] => {}, //~ ERROR expected an array or slice
        _ => {},
    }
    match &x[..] {
        [v] => {},
        _ => {},
    }
    match &x[..] {
        &[v] => {},
        _ => {},
    }
    match x {
        [v] => {}, //~ ERROR expected an array or slice
        _ => {},
    }
    let y = 1i32;
    match &y {
        &v => {},
        _ => {},
    }
    match y {
        &v => {}, //~ ERROR mismatched types
        _ => {},
    }
    match &y {
        v => {},
        _ => {},
    }
    match y {
        v => {},
        _ => {},
    }
    if let [&v] = &x[..] {} //~ ERROR mismatched types
}
