struct Foo(i32);

fn main() {
    let Foo(...) = Foo(0); //~ ERROR unexpected `...`
    let [_, ..., _] = [0, 1]; //~ ERROR unexpected `...`
    let _recovery_witness: () = 0; //~ ERROR mismatched types
}
