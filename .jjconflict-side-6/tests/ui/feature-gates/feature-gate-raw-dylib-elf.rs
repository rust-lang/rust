//@ only-elf
//@ needs-dynamic-linking

#[link(name = "meow", kind = "raw-dylib")] //~ ERROR: link kind `raw-dylib` is unstable on ELF platforms
unsafe extern "C" {
  safe fn meowmeow();
}

fn main() {}
