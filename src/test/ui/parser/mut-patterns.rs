// Can't put mut in non-ident pattern

pub fn main() {
    struct Foo { x: isize }
    let mut Foo { x: x } = Foo { x: 3 }; //~ ERROR: expected one of `:`, `;`, `=`, or `@`, found `{`
}
