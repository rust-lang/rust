// Check that associated paths starting with `<<` are successfully parsed.

fn main() {
    let _: <<A>::B>::C; //~ ERROR cannot find type `A`
    let _ = <<A>::B>::C; //~ ERROR cannot find type `A`
    let <<A>::B>::C; //~ ERROR cannot find type `A`
    let 0 ..= <<A>::B>::C; //~ ERROR cannot find type `A`
    <<A>::B>::C; //~ ERROR cannot find type `A`
}
