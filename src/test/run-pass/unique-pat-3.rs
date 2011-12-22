
tag bar { u(~int); w(int); }

fn main() {
    assert alt u(~10) {
      u(a) {
        log_full(core::error, a);
        *a
      }
      _ { 66 }
    } == 10;
}
