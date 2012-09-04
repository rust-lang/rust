// error-pattern: type cat cannot be dereferenced
struct cat { new() {} }

fn main() {
  let kitty : cat = cat();
  log (error, *kitty);
}