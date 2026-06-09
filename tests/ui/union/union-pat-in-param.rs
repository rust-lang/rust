union U {
    a: &'static i32,
    b: usize,
}

fn fun(U { a }: U) {
    //~^ ERROR access to union field is unsafe
    dbg!(*a);
}

fn main() {
    fun(U { b: 0 });

    let closure = |U { a }| {
        //~^ ERROR access to union field is unsafe
        dbg!(*a);
    };
    closure(U { b: 0 });
}
