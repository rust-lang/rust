fn main() {
    let oa = Some(1);
    let oa2 = Some(1);
    let oa3 = Some(1);
    let v = if let Some(a) = oa {
        Some(&a)
    } else if let Some(a) = oa2 {
        &Some(a) //~ ERROR `if` and `else` have incompatible types [E0308]
    } else if let Some(a) = oa3 {
        &Some(a)
    } else {
        None
    };
    println!("{v:?}");
}
