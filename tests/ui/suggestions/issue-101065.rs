//@ check-fail
//@ run-rustfix

enum FakeResult<T> {
    Ok(T)
}

fn main() {
    let _x = if true {
        FakeResult::Ok(FakeResult::Ok(()))
    } else {
        FakeResult::Ok(()) //~ERROR E0308
    };
}
