fn add(n: int) -> fn@(int) -> int {
      fn@(m: int) -> int { m + n }
}

fn main()
{
      assert add(3)(4) == 7;
      let add3 : fn(int)->int = add(3); //~ ERROR mismatched types
      assert add3(4) == 7;
}