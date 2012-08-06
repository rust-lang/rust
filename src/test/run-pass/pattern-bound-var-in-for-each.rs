// Tests that trans_path checks whether a
// pattern-bound var is an upvar (when translating
// the for-each body)

fn foo(src: uint) {

    match some(src) {
      some(src_id) => {
        for uint::range(0u, 10u) |i| {
            let yyy = src_id;
            assert (yyy == 0u);
        }
      }
      _ => { }
    }
}

fn main() { foo(0u); }
