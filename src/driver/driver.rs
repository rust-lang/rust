#[no_core];
extern mod core(vers = "0.5");

#[cfg(cargo)]
extern mod self(name = "cargo", vers = "0.5");

#[cfg(fuzzer)]
extern mod self(name = "fuzzer", vers = "0.5");

#[cfg(rustdoc)]
extern mod self(name = "rustdoc", vers = "0.5");

#[cfg(rusti)]
extern mod self(name = "rusti", vers = "0.5");

#[cfg(rustc)]
extern mod self(name = "rustc", vers = "0.5");

fn main() { self::main() }