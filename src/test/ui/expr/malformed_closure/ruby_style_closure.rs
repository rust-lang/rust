fn main() {
    let p = Some(45).and_then({
        //~^ expected a `FnOnce<({integer},)>` closure, found `Option<_>`
        |x| println!("doubling {}", x);
        Some(x * 2)
        //~^ ERROR: cannot find value `x` in this scope
    });
}
