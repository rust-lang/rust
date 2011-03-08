fn altlit(int f) -> int {
  alt (f) {
    case (10) {
      log "case 10";
      ret 20;
    }
    case (11) {
      log "case 11";
      ret 22;
    }
  }
}

fn main() {
  check (altlit(10) == 20);
  check (altlit(11) == 22);
}
