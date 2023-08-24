fn get() -> &'static Vec<i32> {
    todo!()
}

fn main() {
    let x = get();
    x.push(1); //~ ERROR cannot borrow `*x` as mutable, as it is behind a `&` reference [E0596]
}
