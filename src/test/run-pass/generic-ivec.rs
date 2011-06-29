// xfail-stage0

fn f[T](@T v) {}
fn main() {
    f(@~[ 1, 2, 3, 4, 5 ]);
}

