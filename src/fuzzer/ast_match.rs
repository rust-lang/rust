use std;
import std::ivec;

fn ivec_equal[T](&T[] v, &T[] u, fn (&T, &T) -> bool element_equality_test) -> bool {
    auto Lv = ivec::len(v);
    if (Lv != ivec::len(u)) {
        ret false;
    }
    auto i = 0u;
    while (i < Lv) {
        if (!element_equality_test(v.(i), u.(i))) {
            ret false;
        }
        i += 1u;
    }
    ret true;
}

fn builtin_equal[T](&T a, &T b) -> bool {
    ret a == b;
}

fn main() {
    // These pass
    assert  builtin_equal(5, 5);
    assert !builtin_equal(5, 4);

    // This passes
    assert !ivec_equal(~[5, 5], ~[5], builtin_equal);

    // These crash
    // https://github.com/graydon/rust/issues/633
    assert !ivec_equal(~[5, 5], ~[5, 4], builtin_equal);
    assert !ivec_equal(~[5, 5], ~[4, 5], builtin_equal);
    assert  ivec_equal(~[5, 5], ~[5, 5], builtin_equal);

    log_err "Pass";
}
