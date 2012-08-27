// Test that a class with an unsendable field can't be
// sent

struct foo {
  let i: int;
  let j: @~str;
  new(i:int, j: @~str) { self.i = i; self.j = j; }
}

fn main() {
  let cat = ~"kitty";
  let po = comm::Port();         //~ ERROR missing `send`
  let ch = comm::Chan(po);       //~ ERROR missing `send`
  comm::send(ch, foo(42, @cat)); //~ ERROR missing `send`
}
