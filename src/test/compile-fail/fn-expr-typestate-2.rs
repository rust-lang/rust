// error-pattern:Unsatisfied precondition
// xfail-stage0

fn main() {
  auto j = (fn () -> int {
        let int i;
        ret i;
    })();
  log_err j;
}
