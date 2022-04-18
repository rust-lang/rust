struct X<'a>(&'a ());
struct S<'a>(&'a dyn Fn(&X) -> &X);
//~^ ERROR missing lifetime specifiers
struct V<'a>(&'a dyn for<'b> Fn(&X) -> &X);
//~^ ERROR missing lifetime specifiers

fn main() {
    let x = S(&|x| {
        println!("hi");
        x
        //~^ ERROR lifetime of reference outlives lifetime of borrowed content... [E0312]
        //~| ERROR mismatched types
    });
    x.0(&X(&()));
}
