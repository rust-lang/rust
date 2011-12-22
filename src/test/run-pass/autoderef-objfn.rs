


// -*- rust -*-
obj clam() {
    fn chowder() { #debug("in clam chowder"); }
}

fn foo(c: @clam) { c.chowder(); }

fn main() { let c: clam = clam(); foo(@c); }
