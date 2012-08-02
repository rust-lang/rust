

fn altlit(f: int) -> int {
    alt check f {
      10 { debug!{"case 10"}; return 20; }
      11 { debug!{"case 11"}; return 22; }
    }
}

fn main() { assert (altlit(10) == 20); assert (altlit(11) == 22); }
