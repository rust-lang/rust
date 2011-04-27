// error-pattern:Unsatisfied precondition

fn main() {
 state obj foo(int x) {
        drop {
          let int baz;
          log(baz);
        }
 }
 fail;
}
