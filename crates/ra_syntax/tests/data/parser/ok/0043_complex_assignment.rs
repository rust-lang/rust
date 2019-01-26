// https://github.com/rust-analyzer/rust-analyzer/issues/674

struct Repr { raw: [u8; 1] }

fn abc() {
    Repr { raw: [0] }.raw[0] = 0;
    Repr{raw:[0]}();
}
