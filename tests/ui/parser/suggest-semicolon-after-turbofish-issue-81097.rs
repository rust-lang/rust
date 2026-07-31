// Regression test for issue #81097: an array length typo inside a turbofish
// should point at the comma and suggest the semicolon separator.

fn main() {
    drop::<[(), 0]>([]); //~ ERROR expected `;` or `]`, found `,`
}
