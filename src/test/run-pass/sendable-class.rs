// Test that a class with only sendable fields can be sent

struct foo {
  i: int,
  j: char,
}

fn foo(i:int, j: char) -> foo {
    foo {
        i: i,
        j: j
    }
}

fn main() {
  let po = comm::Port::<foo>();
  let ch = comm::Chan(po);
  comm::send(ch, foo(42, 'c'));
}