fn main() {
    cfg!(); //~ ERROR macro requires a cfg-pattern
    cfg!(123); //~ ERROR literal in `cfg` predicate value must be a boolean
    cfg!(foo = 123); //~ ERROR literal in `cfg` predicate value must be a string
    cfg!(foo, bar); //~ ERROR expected 1 cfg-pattern
}
