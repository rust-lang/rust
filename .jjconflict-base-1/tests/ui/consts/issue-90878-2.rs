 #![l=|x|[b;x ]] //~ ERROR attribute value must be a literal
//~^ ERROR cannot find attribute `l` in this scope

// notice the space at the start,
// we can't attach any attributes to this file because it needs to be at the start

// this example has been slightly modified (adding ]] at the end), so that it actually works here
// it still produces the same issue though

fn main() {}
