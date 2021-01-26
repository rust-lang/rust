fn main() {
    drop::<[(), 0]>([]); //~ ERROR: expected `;`, found `,`
}
