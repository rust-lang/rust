// Check that associated paths starting with `<<` are successfully parsed.

fn main() {
    let _: <<A>::B>::C; //~ ERROR cannot find type `A` in this scope
    let _ = <<A>::B>::C; //~ ERROR cannot find type `A` in this scope
    let <<A>::B>::C; //~ ERROR cannot find type `A` in this scope
    let 0 ..= <<A>::B>::C; //~ ERROR cannot find type `A` in this scope
    <<A>::B>::C; //~ ERROR cannot find type `A` in this scope
}
