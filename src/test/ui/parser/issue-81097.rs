fn main() {
    drop::<[(), 0]>([]); //~ ERROR: expected one of `;` or `]`, found `,`
}
