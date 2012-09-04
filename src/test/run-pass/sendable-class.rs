// Test that a class with only sendable fields can be sent

struct foo {
  let i: int;
  let j: char;
  new(i:int, j: char) { self.i = i; self.j = j; }
}

fn main() {
  let po = comm::Port::<foo>();
  let ch = comm::Chan(po);
  comm::send(ch, foo(42, 'c'));
}