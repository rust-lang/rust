fn main() {
    let x = some(unsafe::exclusive(false));
    match x {
        some(copy z) => { //~ ERROR copying a noncopyable value
            do z.with |b| { assert !*b; }
        }
        none => fail
    }
}
