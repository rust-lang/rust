fn main() {}

trait Trait1 {
    field_name: String, //~ ERROR expected one of `!` or `::`, found `:`
}

trait Trait2 {
    self: String, //~ ERROR expected one of `!` or `::`, found `:`
}
