fn r#fn() {}

fn main() {
    let r#final = 1;

    // Should correctly suggest variable defined using raw identifier.
    fina; //~ ERROR cannot find value

    // Should correctly suggest function defined using raw identifier.
    f(); //~ ERROR cannot find function
}
