fn main() {
    let x = Some(unsafe::exclusive(false));
    match x {
        Some(copy z) => { //~ ERROR copying a noncopyable value
            do z.with |b| { assert !*b; }
        }
        None => fail
    }
}
