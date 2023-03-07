// min-lldb-version: 310

// compile-flags:-g

// No debugger interaction required: just make sure it compiles without
// crashing.

fn test(a: &Vec<u8>) {
  print!("{}", a.len());
}

pub fn main() {
  let data = vec![];
  test(&data);
}
