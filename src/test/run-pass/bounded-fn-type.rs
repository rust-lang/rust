fn ignore<T>(_x: T) {}

fn main() {
    let f: fn@:Send() = ||();
    ignore(f);
}

