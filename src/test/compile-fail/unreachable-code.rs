// error-pattern:unreachable statement
fn main() {
  loop{}
             // red herring to make sure compilation fails
  log(error, 42 == 'c');
}