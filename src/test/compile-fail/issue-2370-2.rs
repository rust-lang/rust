// error-pattern: type cat cannot be dereferenced
class cat { new() {} }

fn main() {
  let kitty : cat = cat();
  log (error, *kitty);
}