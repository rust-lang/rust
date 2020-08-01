struct X<'a>(&'a ());
struct S<'a>(&'a dyn Fn(&X) -> &X);
//~^ ERROR: missing lifetime specifier
//~| ERROR: missing lifetime specifier

fn main() {
    let x = S(&|x| {
        x
    });
    x.0(&X(&()));
}
