// Type ascription is unstable

fn main() {
    let a = type_ascribe!(10, u8); //~ ERROR use of unstable library feature `type_ascription`: placeholder syntax for type ascription
}
