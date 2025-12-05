fn main() {
    let mut xs: Vec<isize> = vec![];

    for x in &mut xs {
        xs.push(1) //~ ERROR cannot borrow `xs`
    }
}
