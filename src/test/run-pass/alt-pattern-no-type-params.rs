enum maybe<T> { nothing, just(T), }

fn foo(x: maybe<int>) {
    match x { nothing => { error!{"A"}; } just(a) => { error!{"B"}; } }
}

fn main() { }
