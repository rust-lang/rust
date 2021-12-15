// pp-exact

fn main() {}

struct C {
    field: u8,
}

#[allow()]
const C: C =
    C{
      #[cfg(debug_assertions)]
      field: 0,

      #[cfg(not(debug_assertions))]
      field: 1,};
