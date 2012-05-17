// error-pattern: type cat cannot be dereferenced
class cat { new() {} }

fn main() {
  let nyan = cat();
  log (error, *nyan);
}