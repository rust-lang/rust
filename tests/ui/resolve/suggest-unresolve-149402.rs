//@ revisions: stable nightly
//@ [nightly] only-nightly
//@ [stable] only-stable

fn foo() {
    let x = Vec::new();
    x.push(Complete::Item { name: "hello" });
    //~^ ERROR failed to resolve: use of undeclared type `Complete`
}

fn main() {}
