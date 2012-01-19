// error-pattern:Declaration of thpppt shadows
tag ack { thpppt; ffff; }

fn main() {
  let thpppt: int = 42;
  log(debug, thpppt);
}
