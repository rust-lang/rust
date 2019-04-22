fn id<T>(x: T) -> T { x }

fn main() {
    let x = Some(3);
    let y = x.as_ref().unwrap_or(&id(5));  //~ ERROR
    &y;
}
