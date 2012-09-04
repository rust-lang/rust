// error-pattern: type cat cannot be dereferenced
struct cat { new() {} }

fn main() {
  let nyan = cat();
  log (error, *nyan);
}