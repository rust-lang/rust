fn main() {
    cfg!(); //~ ERROR macro requires a cfg-pattern
    cfg!(123); //~ ERROR expected identifier
    cfg!(foo = 123); //~ ERROR literal in `cfg` predicate value must be a string
}
