fn push[T](&mutable vec[mutable? T] v, &T t) {
    v += vec(t);
}

fn main() {
    auto v = @vec(1, 2, 3);
    push[int](*v, 1);
}

