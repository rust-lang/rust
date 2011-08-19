

fn altlit(f: int) -> int {
    alt f { 10 { log "case 10"; ret 20; } 11 { log "case 11"; ret 22; } }
}

fn main() { assert (altlit(10) == 20); assert (altlit(11) == 22); }
