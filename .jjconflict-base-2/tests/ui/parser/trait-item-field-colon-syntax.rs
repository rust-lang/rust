fn main() {}

trait Trait1 {
    field_name: String, //~ ERROR fields are not allowed in trait definitions
}

trait Trait2 {
    self: String, //~ ERROR fields are not allowed in trait definitions
}

trait Trait3 {
    field_name: String //~ ERROR fields are not allowed in trait definitions
}

trait Trait4 {
    self: String //~ ERROR fields are not allowed in trait definitions
}
