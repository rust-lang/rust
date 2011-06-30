resource finish[T](rec(T val, fn(&T) fin) arg) {
    arg.fin(arg.val);
}

fn main() {
    auto box = @mutable 10;
    fn dec_box(&@mutable int i) { *i -= 1; }

    {
        auto i <- finish(rec(val=box, fin=dec_box));
    }
    assert(*box == 9);
}
