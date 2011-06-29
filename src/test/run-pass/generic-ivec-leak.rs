// xfail-stage0

tag wrapper[T] {
    wrapped(T);
}

fn main() {
    auto w = wrapped(~[ 1, 2, 3, 4, 5 ]);
}

