

fn altlit(f: int) -> int {
    match f {
      10 => { debug!("case 10"); return 20; }
      11 => { debug!("case 11"); return 22; }
      _  => fail ~"the impossible happened"
    }
}

fn main() { assert (altlit(10) == 20); assert (altlit(11) == 22); }
