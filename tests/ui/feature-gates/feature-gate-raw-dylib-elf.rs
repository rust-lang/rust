//@ only-elf

#[link(name = "meow", kind = "raw-dylib")] //~ ERROR: link kind `raw-dylib` is unstable on ELF platforms
unsafe extern "C" {
  safe fn meowmeow();
}

fn main() {}
