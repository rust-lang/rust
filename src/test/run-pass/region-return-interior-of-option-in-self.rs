// xfail-test (#3148)

struct cell<T> {
    value: T;
}

struct cells<T> {
    vals: ~[option<cell<T>>];
}

impl<T> &cells<T> {
    fn get(idx: uint) -> &self/T {
        match self.vals[idx] {
          some(ref v) => &v.value,
          none => fail
        }
    }
}

fn main() {}
