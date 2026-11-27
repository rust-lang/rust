//@ known-bug: #157197
static C: &'static usize = &(0 | E);
static E: usize = E;
