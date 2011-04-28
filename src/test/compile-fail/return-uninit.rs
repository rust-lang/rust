// error-pattern: precondition constraint

fn f() -> int {
   let int x;
   ret x;
}

fn main() {
   f();
}