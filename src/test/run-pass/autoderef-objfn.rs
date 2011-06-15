


// -*- rust -*-
obj clam() {
    fn chowder() { log "in clam chowder"; }
}

fn foo(@clam c) { c.chowder(); }

fn main() { let clam c = clam(); foo(@c); }